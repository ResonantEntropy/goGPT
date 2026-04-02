package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strings"
)

// ===========================================================================
//  CONFIGURATION
// ===========================================================================

const (
	Debug      = 1
	nEmbd      = 8
	nHead      = 2
	nLayer     = 1
	blockSize  = 8
	maxDocs    = 30000
	initStd    = 0.02
	numSteps   = 1000
	lr         = 0.01
	beta1      = 0.85
	beta2      = 0.99
	epsAdam    = 1e-8
	numSamples = 10
	temp       = 0.7
)

var headDim = nEmbd / nHead

// ===========================================================================
//  AUTOGRAD ENGINE
// ===========================================================================

var (
	valuesData       = make(map[int]float64)
	valuesGrad       = make(map[int]float64)
	valuesChildren   = make(map[int][]int)
	valuesLocalGrads = make(map[int][]float64)
	valueCounter     = 0
)

func createValue(data float64, children []int, localGrads []float64) int {
	id := valueCounter
	valuesData[id] = data
	valuesGrad[id] = 0
	valuesChildren[id] = children
	valuesLocalGrads[id] = localGrads
	valueCounter++
	return id
}

// Arithmetic Helpers
func add(a, b int) int {
	return createValue(valuesData[a]+valuesData[b], []int{a, b}, []float64{1.0, 1.0})
}
func mul(a, b int) int {
	return createValue(valuesData[a]*valuesData[b], []int{a, b}, []float64{valuesData[b], valuesData[a]})
}
func neg(a int) int    { return mul(a, createValue(-1.0, nil, nil)) }
func sub(a, b int) int { return add(a, neg(b)) }
func pow(a int, p float64) int {
	return createValue(math.Pow(valuesData[a], p), []int{a}, []float64{p * math.Pow(valuesData[a], p-1)})
}
func div(a, b int) int { return mul(a, pow(b, -1.0)) }
func exp(a int) int {
	res := math.Exp(valuesData[a])
	return createValue(res, []int{a}, []float64{res})
}
func log(a int) int {
	return createValue(math.Log(valuesData[a]), []int{a}, []float64{1.0 / valuesData[a]})
}
func relu(a int) int {
	if valuesData[a] < 0 {
		return createValue(0, []int{a}, []float64{0})
	}
	return createValue(valuesData[a], []int{a}, []float64{1})
}

func backward(rootID int) {
	topo := []int{}
	visited := make(map[int]bool)
	var buildTopo func(int)
	buildTopo = func(v int) {
		if !visited[v] {
			visited[v] = true
			for _, child := range valuesChildren[v] {
				buildTopo(child)
			}
			topo = append(topo, v)
		}
	}
	buildTopo(rootID)
	valuesGrad[rootID] = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		for j, child := range valuesChildren[v] {
			valuesGrad[child] += valuesLocalGrads[v][j] * valuesGrad[v]
		}
	}
}

// ===========================================================================
//  KV-CACHE & ATTENTION
// ===========================================================================

type KVCache struct {
	// [layer][timestep][headDim*nHead]
	Keys   [][][]int
	Values [][][]int
}

func newKVCache() *KVCache {
	return &KVCache{
		Keys:   make([][][]int, nLayer),
		Values: make([][][]int, nLayer),
	}
}

func rmsnorm(x []int) []int {
	var sumSq float64
	for _, id := range x {
		sumSq += valuesData[id] * valuesData[id]
	}
	scale := createValue(1.0/math.Sqrt(sumSq/float64(len(x))+1e-5), nil, nil)
	res := make([]int, len(x))
	for i, id := range x {
		res[i] = mul(id, scale)
	}
	return res
}

func softmax(logits []int) []int {
	maxVal := -1e9
	for _, id := range logits {
		if valuesData[id] > maxVal {
			maxVal = valuesData[id]
		}
	}
	maxID := createValue(maxVal, nil, nil)
	exps := make([]int, len(logits))
	sumExp := createValue(0, nil, nil)
	for i, id := range logits {
		exps[i] = exp(sub(id, maxID))
		sumExp = add(sumExp, exps[i])
	}
	probs := make([]int, len(logits))
	for i, e := range exps {
		probs[i] = div(e, sumExp)
	}
	return probs
}

type StateDict map[string]int

func linear(x []int, prefix string, nout int, state StateDict) []int {
	res := make([]int, nout)
	for o := 0; o < nout; o++ {
		sum := createValue(0, nil, nil)
		for i := 0; i < len(x); i++ {
			sum = add(sum, mul(state[fmt.Sprintf("%s_%d_%d", prefix, o, i)], x[i]))
		}
		res[o] = sum
	}
	return res
}

func gpt(tokenID, posID int, state StateDict, vocabSize int, cache *KVCache) []int {
	x := make([]int, nEmbd)
	for j := 0; j < nEmbd; j++ {
		x[j] = add(state[fmt.Sprintf("wte_%d_%d", tokenID, j)], state[fmt.Sprintf("wpe_%d_%d", posID, j)])
	}
	x = rmsnorm(x)

	for li := 0; li < nLayer; li++ {
		residual := x
		x = rmsnorm(x)

		q := linear(x, fmt.Sprintf("layer%d.attn_wq", li), nEmbd, state)
		k := linear(x, fmt.Sprintf("layer%d.attn_wk", li), nEmbd, state)
		v := linear(x, fmt.Sprintf("layer%d.attn_wv", li), nEmbd, state)

		// Update Cache
		cache.Keys[li] = append(cache.Keys[li], k)
		cache.Values[li] = append(cache.Values[li], v)

		xAttn := make([]int, 0, nEmbd)
		for h := 0; h < nHead; h++ {
			hs := h * headDim
			qh := q[hs : hs+headDim]

			attnLogits := make([]int, len(cache.Keys[li]))
			for t := 0; t < len(cache.Keys[li]); t++ {
				kh := cache.Keys[li][t][hs : hs+headDim]
				dot := createValue(0, nil, nil)
				for j := 0; j < headDim; j++ {
					dot = add(dot, mul(qh[j], kh[j]))
				}
				attnLogits[t] = div(dot, createValue(math.Sqrt(float64(headDim)), nil, nil))
			}

			aw := softmax(attnLogits)
			headOut := make([]int, headDim)
			for j := 0; j < headDim; j++ {
				sum := createValue(0, nil, nil)
				for t := 0; t < len(cache.Values[li]); t++ {
					sum = add(sum, mul(aw[t], cache.Values[li][t][hs : hs+headDim][j]))
				}
				headOut[j] = sum
			}
			xAttn = append(xAttn, headOut...)
		}

		x = addVec(linear(xAttn, fmt.Sprintf("layer%d.attn_wo", li), nEmbd, state), residual)

		residualMLP := x
		x = rmsnorm(x)
		x = linear(x, fmt.Sprintf("layer%d.mlp_fc1", li), 4*nEmbd, state)
		for i := range x {
			x[i] = relu(x[i])
		}
		x = addVec(linear(x, fmt.Sprintf("layer%d.mlp_fc2", li), nEmbd, state), residualMLP)
	}
	return linear(x, "lm_head", vocabSize, state)
}

func addVec(a, b []int) []int {
	res := make([]int, len(a))
	for i := range a {
		res[i] = add(a[i], b[i])
	}
	return res
}

// ===========================================================================
//  TRAINING & INFERENCE
// ===========================================================================

func main() {
	// Prep Data
	if _, err := os.Stat("input.txt"); os.IsNotExist(err) {
		resp, _ := http.Get("https://raw.githubusercontent.com/ResonantEntropy/goGPT/refs/heads/main/input.txt")
		f, _ := os.Create("input.txt")
		f.ReadFrom(resp.Body)
		f.Close()
		resp.Body.Close()
	}
	file, _ := os.Open("input.txt")
	scanner := bufio.NewScanner(file)
	var docs []string
	charSet := make(map[rune]bool)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
			for _, r := range line {
				charSet[r] = true
			}
		}
	}
	file.Close()
	rand.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })

	var uchars []rune
	for r := range charSet {
		uchars = append(uchars, r)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	charToID := make(map[rune]int)
	for i, r := range uchars {
		charToID[r] = i
	}
	BOS := len(uchars)
	vocabSize := BOS + 1

	// Initialise
	state := make(StateDict)
	params := []int{}
	initP := func(n string, r, c int) {
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				id := createValue(rand.NormFloat64()*initStd, nil, nil)
				state[fmt.Sprintf("%s_%d_%d", n, i, j)] = id
				params = append(params, id)
			}
		}
	}
	initP("wte", vocabSize, nEmbd)
	initP("lm_head", vocabSize, nEmbd)
	initP("wpe", blockSize, nEmbd)
	for l := 0; l < nLayer; l++ {
		initP(fmt.Sprintf("layer%d.attn_wq", l), nEmbd, nEmbd)
		initP(fmt.Sprintf("layer%d.attn_wk", l), nEmbd, nEmbd)
		initP(fmt.Sprintf("layer%d.attn_wv", l), nEmbd, nEmbd)
		initP(fmt.Sprintf("layer%d.attn_wo", l), nEmbd, nEmbd)
		initP(fmt.Sprintf("layer%d.mlp_fc1", l), 4*nEmbd, nEmbd)
		initP(fmt.Sprintf("layer%d.mlp_fc2", l), nEmbd, 4*nEmbd)
	}

	firstTmp := valueCounter
	mAdam, vAdam := make(map[int]float64), make(map[int]float64)

	// Training Loop
	fmt.Printf("Training %d parameters...\n", len(params))
	for step := 0; step < numSteps; step++ {
		doc := docs[step%len(docs)]
		tokens := []int{BOS}
		for _, r := range doc {
			tokens = append(tokens, charToID[r])
		}
		tokens = append(tokens, BOS)

		n := len(tokens) - 1
		if n > blockSize {
			n = blockSize
		}

		cache := newKVCache()
		losses := []int{}
		for pos := 0; pos < n; pos++ {
			logits := gpt(tokens[pos], pos, state, vocabSize, cache)
			probs := softmax(logits)
			losses = append(losses, neg(log(probs[tokens[pos+1]])))
		}

		sumL := createValue(0, nil, nil)
		for _, l := range losses {
			sumL = add(sumL, l)
		}
		loss := div(sumL, createValue(float64(n), nil, nil))

		backward(loss)

		// Optimizer Update
		for _, p := range params {
			g := valuesGrad[p]
			mAdam[p] = beta1*mAdam[p] + (1-beta1)*g
			vAdam[p] = beta2*vAdam[p] + (1-beta2)*g*g
			valuesData[p] -= lr * (mAdam[p] / (1 - math.Pow(beta1, float64(step+1)))) / (math.Sqrt(vAdam[p]/(1-math.Pow(beta2, float64(step+1)))) + epsAdam)
			valuesGrad[p] = 0
		}

		if step%10 == 0 {
			fmt.Printf("Step %d | Loss: %.4f\n", step, valuesData[loss])
		}
		for i := firstTmp; i < valueCounter; i++ {
			delete(valuesData, i)
			delete(valuesGrad, i)
			delete(valuesChildren, i)
			delete(valuesLocalGrads, i)
		}
		valueCounter = firstTmp
	}

	// KV-Cached Inference
	fmt.Println("\n--- KV-Cached Inference ---")
	for s := 0; s < numSamples; s++ {
		cache := newKVCache()
		tokenID := BOS
		sample := ""
		for pos := 0; pos < blockSize; pos++ {
			logits := gpt(tokenID, pos, state, vocabSize, cache)
			probs := softmax(logits)

			r, cum := rand.Float64(), 0.0
			chosen := BOS
			for i, pID := range probs {
				cum += valuesData[pID]
				if r < cum {
					chosen = i
					break
				}
			}
			if chosen == BOS {
				break
			}
			sample += string(uchars[chosen])
			tokenID = chosen
		}
		fmt.Println(sample)
		for i := firstTmp; i < valueCounter; i++ {
			delete(valuesData, i)
		}
		valueCounter = firstTmp
	}
}
