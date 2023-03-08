package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"os"

	//"strconv"

	"encoding/xml"

	//"github.com/beevik/etree"

	//"github.com/tamerh/goxml"

	vision "cloud.google.com/go/vision/apiv1"
	//"golang.org/x/net/context"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"

	"gocv.io/x/gocv"
	//"google.golang.org/api/vision/v1"
	//"cloud.google.com/go/vision/apiv1"
)

func rotateImage(img gocv.Mat, angle float64) gocv.Mat {
	size := image.Point{img.Cols(), img.Rows()}
	center := image.Point{size.X / 2, size.Y / 2}
	mat := gocv.GetRotationMatrix2D(center, angle, 1.0)
	rotImg := gocv.NewMat()
	gocv.WarpAffine(img, &rotImg, mat, size)
	return rotImg
}

type Annotation struct {
	XMLName xml.Name `xml:"annotation"`
	Objects []Object `xml:"object"`
}

type Object struct {
	Name   string `xml:"name"`
	Bndbox Bndbox `xml:"bndbox"`
}

type Bndbox struct {
	Xmin int `xml:"xmin"`
	Ymin int `xml:"ymin"`
	Xmax int `xml:"xmax"`
	Ymax int `xml:"ymax"`
}

func main() {

	os.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "./data/vision.json")
	// Load image
	imgFile, err := os.Open("./number.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer imgFile.Close()

	//imgdcode, _, err := image.Decode(imgFile)
	//outputFile := "./img/output_file"

	/* outFile, err := os.Create("./img/output_file.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer outFile.Close() */

	img := gocv.IMRead("./number.jpg", gocv.IMReadColor)
	gocv.IMWrite("img/output_org.jpg", img)

	// Convert to grayscale
	grayImg := gocv.NewMat()
	gocv.CvtColor(img, &grayImg, gocv.ColorBGRToGray)

	// Write grayscale image to disk
	gocv.IMWrite("img/output_gray.jpg", grayImg)

	// Blur image
	imgAvg := gocv.NewMat()
	kSize := image.Pt(8, 8)
	gocv.Blur(grayImg, &imgAvg, kSize)

	// Write blurred image to disk
	gocv.IMWrite("img/output_avg.jpg", imgAvg)

	// Binarize image
	binarized := gocv.NewMat()
	gocv.Threshold(imgAvg, &binarized, 100, 255, gocv.ThresholdBinary)

	// Write binarized image to disk
	gocv.IMWrite("img/output_bin.jpg", binarized)

	contours := gocv.FindContours(binarized, gocv.RetrievalExternal, gocv.ChainApproxSimple)

	var maxArea float64 = 0
	var maxContourIdx int = 0

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)
		if area > maxArea {
			maxArea = area
			maxContourIdx = i
			//fmt.Printf("Maximum contour area: %f\n", maxArea)

		}

	}

	lineColor := color.RGBA{0, 255, 0, 0}
	thickness := 30 // fill the contour
	gocv.DrawContours(&img, contours, maxContourIdx, lineColor, thickness)

	gocv.IMWrite("./img/output_con.jpg", img)

	largestContour := contours.At(maxContourIdx)
	//fmt.Printf("largest contour:=%d/n", largestContour)
	epsilon := 0.1 * gocv.ArcLength(largestContour, true)
	approx := gocv.ApproxPolyDP(largestContour, epsilon, true)
	cardImgWidth := 1180
	//fmt.Printf("width:=%d/n", cardImgWidth)
	cardImgHeight := int(math.Round(float64(cardImgWidth) * (5.4 / 8.56)))
	//fmt.Printf("height:=%d/n", cardImgHeight)

	//creates new slice
	src := make([]float32, 0)
	//gets size of approx
	size := approx.Size()
	for i := 0; i < size; i++ {
		//iterates and using AT we get each x & y axis
		point := approx.At(i)
		//then append all of the x&y cordinates to the slice through append
		src = append(src, float32(point.X), float32(point.Y))
	}
	//fmt.Printf("src:=%f/n", src)
	//initialise a slice with 4 points as 4 corners
	dst := []float32{
		0, 0,
		0, float32(cardImgHeight),
		float32(cardImgWidth), float32(cardImgHeight),
		float32(cardImgWidth), 0,
	}
	//fmt.Printf("dst:=%f/n", dst)
	//we have to find no of rows so,divide by 2 bcz there is x and y in slice
	//col=1
	//type of matrix
	srcMat := gocv.NewMatWithSize(len(src)/2, 1, gocv.MatTypeCV32FC2)
	for i := 0; i < len(src); i += 2 {
		//find row index ,col index--->forx=0 & y=1
		srcMat.SetFloatAt(i/2, 0, src[i])
		srcMat.SetFloatAt(i/2, 1, src[i+1])
	}

	dstMat := gocv.NewMatWithSize(len(dst)/2, 1, gocv.MatTypeCV32FC2)
	for i := 0; i < len(dst); i += 2 {
		dstMat.SetFloatAt(i/2, 0, dst[i])
		dstMat.SetFloatAt(i/2, 1, dst[i+1])
	}
	//assigning the matrixs to a pointvector
	//bcz for prespecvtrnsfrm we have to use pointvector
	srcPoints := gocv.NewPointVectorFromMat(srcMat)
	dstPoints := gocv.NewPointVectorFromMat(dstMat)

	projectMatrix := gocv.GetPerspectiveTransform(srcPoints, dstPoints)
	//fmt.Printf("matrix:=%v/n", projectMatrix)

	//overlined the previous image ,so read again
	imgOrg := gocv.IMRead("./number.jpg", gocv.IMReadColor)
	defer imgOrg.Close()
	//create a variable which is mat
	transformed := gocv.NewMat()
	gocv.WarpPerspective(imgOrg, &transformed, projectMatrix, image.Point{X: cardImgWidth, Y: cardImgHeight})

	// Save output image
	gocv.IMWrite("./img/output_tra.jpg", transformed)

	/* func rotateImage(img, angle float64) gocv.Mat {
		size := image.Point{img.Cols(), img.Rows()}
		center := image.Point{size.X / 2, size.Y / 2}
		mat := gocv.GetRotationMatrix2D(center, angle, 1.0)
		var rotImg gocv.Mat
		gocv.WarpAffine(img, &rotImg, mat, size, gocv.InterpolationCubic)
		return rotImg
	} */

	/* angle := 1.0
	imgRot := rotateImg(transformed, angle)
	gocv.IMWrite(outputFile, imgRot)
	*/
	output_file := "./img/output_tmp.jpg"
	img_rot := rotateImage(transformed, 1.0)
	gocv.IMWrite(output_file, img_rot)

	ctx := context.Background()

	// Create a new client
	client, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	// Read image file
	imageData, err := ioutil.ReadFile(output_file)
	if err != nil {
		log.Fatalf("Failed to read image file: %v", err)
	}

	// Create a new text detection request
	image := &pb.Image{
		Content: imageData,
	}
	features := []*pb.Feature{
		{Type: pb.Feature_TEXT_DETECTION},
	}
	// Create image context with language hints
	imageContext := &pb.ImageContext{
		LanguageHints: []string{"ja"},
	}
	req := &pb.AnnotateImageRequest{
		Image:        image,
		Features:     features,
		ImageContext: imageContext,
	}

	// Send the text detection request
	res, err := client.AnnotateImage(ctx, req)
	if err != nil {
		log.Fatalf("Failed to detect text: %v", err)
	}
	type TextInfo struct {
		Text    string
		XCenter float64
		YCenter float64
	}

	// Process the text detection response
	// define textInfos as a slice of a custom struct that contains fields for the text, xcenter, and ycenter:
	textInfos := []TextInfo{}
	//textInfos := [][]interface{}{}
	if annotation := res.GetFullTextAnnotation(); annotation != nil {
		for _, page := range annotation.GetPages() {
			for _, block := range page.GetBlocks() {
				for _, paragraph := range block.GetParagraphs() {
					for _, word := range paragraph.GetWords() {
						for _, symbol := range word.GetSymbols() {
							boundingBox := symbol.GetBoundingBox()
							xmin := boundingBox.GetVertices()[0].GetX()
							ymin := boundingBox.GetVertices()[0].GetY()
							xmax := boundingBox.GetVertices()[2].GetX()
							ymax := boundingBox.GetVertices()[2].GetY()
							xcenter := float64(xmin+xmax) / 2
							ycenter := float64(ymin+ymax) / 2
							text := symbol.GetText()
							//textInfo := TextInfo{text, xcenter, ycenter}
							//textInfos = append(textInfos, textInfo)
							textInfos = append(textInfos, TextInfo{text, xcenter, ycenter})
							//textInfos = append(textInfos, []interface{}{text, xcenter, ycenter})
						}
					}
				}
			}
		}
	}
	// Read the XML file
	xmlFile, err := ioutil.ReadFile("data/mynumber.xml")
	if err != nil {
		panic(err)
	}

	// Parse the XML data
	var annotation Annotation
	err = xml.Unmarshal(xmlFile, &annotation)
	if err != nil {
		panic(err)
	}

	// Extract the relevant information
	//creates an empty map with keys of type string and values of type string
	//used to store the detected text for each object in the input image
	resultDict := make(map[string]string)
	for _, obj := range annotation.Objects {
		name := obj.Name
		xmin := obj.Bndbox.Xmin
		ymin := obj.Bndbox.Ymin
		xmax := obj.Bndbox.Xmax
		ymax := obj.Bndbox.Ymax
		texts := ""
		for _, textInfo := range textInfos {
			if float64(xmin) <= textInfo.XCenter && textInfo.XCenter <= float64(xmax) && float64(ymin) <= textInfo.YCenter && textInfo.YCenter <= float64(ymax) {
				texts += textInfo.Text
			}
		}
		resultDict[name] = texts
	}

	// Print the results
	for k, v := range resultDict {
		fmt.Printf("%s : %s\n", k, v)
	}

}
