using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

class Program
{
    static void Main(string[] args)
    {
        // Read the input image
        Mat src = CvInvoke.Imread("component2 copy.jpg");

        // Convert image to gray and blur it
        Mat src_gray = new Mat();
        CvInvoke.CvtColor(src, src_gray, ColorConversion.Bgr2Gray);
        CvInvoke.GaussianBlur(src_gray, src_gray, new Size(3, 3), 0);

        // Create Window
        string source_window = "Source";
        CvInvoke.NamedWindow(source_window, NamedWindowType.Normal);
        CvInvoke.Imshow(source_window, src);

        int max_thresh = 255;
        int thresh = 100; // Lower initial threshold

        ThreshCallback(src_gray, thresh);

        CvInvoke.WaitKey(0);
        CvInvoke.DestroyAllWindows();
    }

    static void DrawRedGrid(Mat img)
    {
        MCvScalar lineColor = new MCvScalar(0, 0, 150);
        int lineThickness = 1;

        // Define the spacing between grid lines (in pixels) corresponding to 1 cm
        int gridSpacing = 15;

        // Draw vertical grid lines
        for (int x = 0; x < img.Cols; x += gridSpacing)
            CvInvoke.Line(img, new Point(x, 0), new Point(x, img.Rows), lineColor, lineThickness);

        // Draw horizontal grid lines
        for (int y = 0; y < img.Rows; y += gridSpacing)
            CvInvoke.Line(img, new Point(0, y), new Point(img.Cols, y), lineColor, lineThickness);
    }

    static void ThreshCallback(Mat src_gray, int val)
    {
        int threshold = val;

        // Detect edges using Canny
        Mat canny_output = new Mat();
        CvInvoke.Canny(src_gray, canny_output, threshold, threshold * 2);

        // Find contours
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(canny_output, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        // Create a black image to draw polygons and grid
        Mat drawing = new Mat(src_gray.Size, DepthType.Cv8U, 3);

        // Draw polygons only for contours
        for (int i = 0; i < contours.Size; i++)
        {
            double area = CvInvoke.ContourArea(contours[i]);
            if (area > 300)
            {
                double epsilon = 0.01 * CvInvoke.ArcLength(contours[i], true);
                VectorOfPoint approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contours[i], approx, epsilon, true);

                // Filter by aspect ratio - polygons can't be too thin
                Rectangle boundingRect = CvInvoke.BoundingRectangle(approx);
                double aspectRatio = (double)boundingRect.Width / boundingRect.Height;

                // Filter by elongation = essentially get rid of the wire paths
                double elongationThreshold = 1.53;
                if (0.1 < aspectRatio && aspectRatio < elongationThreshold)
                    CvInvoke.DrawContours(drawing, new VectorOfVectorOfPoint { approx }, 0, new MCvScalar(255, 255, 255), 2);
            }
        }

        // Draw red grid
        DrawRedGrid(drawing);

        // Show the result
        
        CvInvoke.Imwrite("processed.jpg", drawing);
    }
}
