// Command timestral times Timestral forecasts (ONNX export, see export.py) on a GoMLX backend,
// the way we run it in production: inference only, one (batch, context) input shape compiled once,
// batch 2 = context and -context (the forecast is averaged with the mirrored one).
//
//	GOEXPERIMENT=simd go run . -dir model-28m -runs 5
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"syscall"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/compute/gobackend" // registers the pure Go backend "go"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

// meta is written by export.py.
type meta struct {
	Model        string    `json:"model"`
	ContextSteps int       `json:"context_steps"`
	HorizonSteps int       `json:"horizon_steps"`
	Quantiles    []float64 `json:"quantile_levels"`
}

func main() {
	obs := flag.String("obs", "", "Optional observation label for the rows printed")
	dir := flag.String("dir", "model-28m", "export.py output: timestral.onnx, meta.json, context.f32, reference.f32")
	backendConfig := flag.String("backend", "go", "GoMLX backend config")
	runs := flag.Int("runs", 5, "timed runs after the first (warm-up) one")
	flag.Parse()

	var m meta
	must(json.Unmarshal(must1(os.ReadFile(filepath.Join(*dir, "meta.json"))), &m))
	context := readF32(filepath.Join(*dir, "context.f32"))
	var reference []float32 // (2, horizon, quantiles) from PyTorch; optional
	if _, err := os.Stat(filepath.Join(*dir, "reference.f32")); err == nil {
		reference = readF32(filepath.Join(*dir, "reference.f32"))
	}
	const batch = 2
	input := make([]float32, batch*m.ContextSteps)
	copy(input, context)
	for i, v := range context {
		input[m.ContextSteps+i] = -v
	}
	fmt.Printf("%s: context %d, horizon %d, %d quantiles, batch %d, backend %q, GOMAXPROCS %d\n",
		m.Model, m.ContextSteps, m.HorizonSteps, len(m.Quantiles), batch, *backendConfig, runtime.GOMAXPROCS(0))

	t0 := time.Now()
	backend := must1(compute.NewWithConfig(*backendConfig))
	onnxModel := must1(parser.ParseFile(filepath.Join(*dir, "timestral.onnx")))
	// The weights become constants of the compiled graph, so the store is not needed afterwards.
	store := mlmodel.NewStore().WithVariableAsConst(true)
	must(onnxModel.VariablesToScope(store.RootScope()))
	exec := must1(mlmodel.NewExec(backend, store, func(scope *mlmodel.Scope, x *graph.Node) *graph.Node {
		return onnxModel.CallGraph(scope, x.Graph(), map[string]*graph.Node{"context": x}, "quantiles")[0]
	}))
	// Only this input shape is used: compile it, then drop the parsed ONNX model and the variables.
	must1(exec.Compile(shapes.Make(dtypes.Float32, batch, m.ContextSteps)))
	onnxModel = nil
	store.Finalize()
	debug.FreeOSMemory()
	fmt.Printf("parse + compile: %.2fs, max RSS %.2f GB\n", time.Since(t0).Seconds(), maxRSS())

	fmt.Println()
	fmt.Println("| Obs | Model | Run | Wall | CPU Time | Cores | Max Abs Diff vs PyTorch | Max RSS |")
	fmt.Println("| --- | --- | :--- | ---: | ---: | ---: | ---: | ---: |")
	var (
		bestDt    time.Duration
		bestDc    time.Duration
		bestCheck string
		bestRSS   float64
	)
	for r := range *runs + 1 {
		t, c := time.Now(), cpuTime()
		x := tensors.FromFlatDataAndDimensions(input, batch, m.ContextSteps)
		y := must1(exec.Call1(x))
		got := must1(tensors.CopyFlatData[float32](y))
		dt, dc := time.Since(t), cpuTime()-c
		x.FinalizeAll()
		y.FinalizeAll()
		check := "-"
		if len(reference) == len(got) {
			diff := 0.0
			for i, v := range got {
				diff = max(diff, math.Abs(float64(v)-float64(reference[i])))
			}
			check = fmt.Sprintf("%.2g", diff)
		}
		rss := maxRSS()
		if r >= 1 && (bestDt == 0 || dt < bestDt) {
			bestDt = dt
			bestDc = dc
			bestCheck = check
			bestRSS = rss
		}
		label := fmt.Sprintf("run %d", r)
		if r == 0 {
			label = "warm-up"
		}
		fmt.Printf("| %s | %s | %-7s | %.3fs | %.3fs | %.1f | %s | %.2f GB |\n",
			*obs, *dir, label, dt.Seconds(), dc.Seconds(), dc.Seconds()/dt.Seconds(), check, rss)
	}
	if *runs >= 1 {
		fmt.Printf("| %s | %s | %-7s | %.3fs | %.3fs | %.1f | %s | %.2f GB |\n",
			*obs, *dir, "best", bestDt.Seconds(), bestDc.Seconds(), bestDc.Seconds()/bestDt.Seconds(), bestCheck, bestRSS)
	}
}

func readF32(path string) []float32 {
	b := must1(os.ReadFile(path))
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func maxRSS() float64 {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return float64(ru.Maxrss) / 1e9 // bytes
	}
	return float64(ru.Maxrss) * 1024 / 1e9 // KiB
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func must1[T any](v T, err error) T {
	must(err)
	return v
}
