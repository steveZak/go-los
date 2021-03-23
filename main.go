package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	pigo "github.com/esimov/pigo/core"
	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"github.com/galeone/tfgo/image"
	"github.com/galeone/tfgo/image/filter"
	"github.com/galeone/tfgo/image/padding"
)

func main() {
	fmt.Println("golos")
	// pigoDetection()
	createNN()
}

func createNN() {
	//
	// root := tg.NewRoot()
	// A := tg.NewTensor(root, tg.Const(root, [2][2]int32{{1, 2}, {-1, -2}}))
	// x := tg.NewTensor(root, tg.Const(root, [2][1]int64{{10}, {100}}))
	// b := tg.NewTensor(root, tg.Const(root, [2][1]int32{{-10}, {10}}))
	// Y := A.MatMul(x.Output).Add(b.Output)
	// // Please note that Y is just a pointer to A!

	// // If we want to create a different node in the graph, we have to clone Y
	// // or equivalently A
	// Z := A.Clone()
	// results := tg.Exec(root, []tf.Output{Y.Output, Z.Output}, nil, &tf.SessionOptions{})
	// fmt.Println("Y: ", results[0].Value(), "Z: ", results[1].Value())
	// fmt.Println("Y == A", Y == A) // ==> true
	// fmt.Println("Z == A", Z == A) // ==> false
	root := tg.NewRoot()
	grayImg := image.Read(root, "/home/steve/go/src/go-los/test_images/test.jpg", 1)
	grayImg = grayImg.Scale(0, 255)

	// Edge detection using sobel filter: convolution
	Gx := grayImg.Clone().Convolve(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	Gy := grayImg.Clone().Convolve(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	convoluteEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

	Gx = grayImg.Clone().Correlate(filter.SobelX(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	Gy = grayImg.Clone().Correlate(filter.SobelY(root), image.Stride{X: 1, Y: 1}, padding.SAME)
	correlateEdges := image.NewImage(root.SubScope("edge"), Gx.Square().Add(Gy.Square().Value()).Sqrt().Value()).EncodeJPEG()

	results := tg.Exec(root, []tf.Output{convoluteEdges, correlateEdges}, nil, &tf.SessionOptions{})

	file, _ := os.Create("convolved.png")
	file.WriteString(results[0].Value().(string))
	file.Close()

	file, _ = os.Create("correlated.png")
	file.WriteString(results[1].Value().(string))
	file.Close()
	// feed image
	// CNN + RNN -> phoneme probability for each frame
	// NLP prediction for future words, where to break each word up?
}

func pigoDetection() {
	cascadeFile, err := ioutil.ReadFile("facefinder.bin")
	// haarcascade_frontalface_default.xml
	if err != nil {
		log.Fatalf("Error reading the cascade file: %v", err)
	}

	src, err := pigo.GetImage("test_images/test.jpg")
	// ImgToNRGBA(img)
	if err != nil {
		log.Fatalf("Cannot open the image file: %v", err)
	}

	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,

		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	pigo := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := pigo.Unpack(cascadeFile)
	if err != nil {
		log.Fatalf("Error reading the cascade file: %s", err)
	}

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	dets := classifier.RunCascade(cParams, angle)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = classifier.ClusterDetections(dets, 0.2)
	fmt.Println(dets)
	// classifier.
}
