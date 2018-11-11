package com.emaraic.utils;

import com.emaraic.utils.NonMaxSuppression;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.LoggerFactory;

/**
 * Created on Jun 21, 2018 , 4:02:27 PM
 *
 * @author Taha Emara 
 * Email : taha@emaraic.com 
 * Website: http://www.emaraic.com
 */
public class YoloModel {

    private static final String[] CLASSES = {"Rubik Cube"};

    private final int IMAGE_INPUT_W = 416;
    private final int IMAGE_INPUT_H = 416;
    private final int CHANNELS = 3;
    private final int GRID_W = 13;
    private final int GRID_H = 13;
    private final String MODEL_PATH = "model.data";
    private static ComputationGraph NETWORK;
    //private final double DETECTION_THRESHOLD = .5;

    private static final org.slf4j.Logger log = LoggerFactory.getLogger(YoloModel.class);

    public YoloModel() {
        String pathtoexe = System.getProperty("user.dir");
        File net = new File(pathtoexe, "model.data");
        boolean modelexists = net.exists() && !net.isDirectory();

        if (modelexists) {
            try {
                NETWORK = ModelSerializer.restoreComputationGraph(MODEL_PATH);
                //System.out.println(NETWORK.summary());
            } catch (IOException ex) {
                log.error(ex.getMessage());
            }
        } else {
            log.error("Can't find model file \"model.data\"\n"
                    + "Please Train the dataset first to provide the model file");
        }

    }

    private void drawBoxes(Mat image, List<DetectedObject> objects) {
        for (DetectedObject obj : objects) {
            double[] xy1 = obj.getTopLeftXY();
            double[] xy2 = obj.getBottomRightXY();
            int predictedClass = obj.getPredictedClass();
            System.out.println("Predicted class " + CLASSES[predictedClass]);
            int x1 = (int) Math.round(IMAGE_INPUT_W * xy1[0] / GRID_W);
            int y1 = (int) Math.round(IMAGE_INPUT_H * xy1[1] / GRID_H);
            int x2 = (int) Math.round(IMAGE_INPUT_W * xy2[0] / GRID_W);
            int y2 = (int) Math.round(IMAGE_INPUT_H * xy2[1] / GRID_H);
            rectangle(image, new opencv_core.Point(x1, y1), new opencv_core.Point(x2, y2), opencv_core.Scalar.RED);
            putText(image, CLASSES[predictedClass], new opencv_core.Point(x1 + 2, y2 - 2), 1, .8, opencv_core.Scalar.RED);

        }
    }

    public void detectRubixCube(Mat image, double detectionthreshold) {
        Yolo2OutputLayer yout = (Yolo2OutputLayer) NETWORK.getOutputLayer(0);
        NativeImageLoader loader = new NativeImageLoader(IMAGE_INPUT_W, IMAGE_INPUT_H, CHANNELS);//, new ColorConversionTransform(COLOR_BGR2RGB)
        INDArray ds = null;
        try {
            ds = loader.asMatrix(image);
        } catch (IOException ex) {
            log.error(ex.getMessage());
        }
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(ds);
        INDArray results = NETWORK.outputSingle(ds);
        List<DetectedObject> objs = yout.getPredictedObjects(results, detectionthreshold);
        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);
        drawBoxes(image, objects);//use objs to see the use of the NonMax Suppression algorithm
    }

    public void detectRubixCube(INDArray inputimage, Mat outputimage, double detectionthreshold) {
        Yolo2OutputLayer yout = (Yolo2OutputLayer) NETWORK.getOutputLayer(0);
        INDArray results = NETWORK.outputSingle(inputimage);
        List<DetectedObject> objs = yout.getPredictedObjects(results, detectionthreshold);
        List<DetectedObject> objects = NonMaxSuppression.getObjects(objs);
        putText(outputimage, "Hit any key in your keyboard to test the next image..", new opencv_core.Point(10, 25), 1,.9, opencv_core.Scalar.BLACK);
        drawBoxes(outputimage, objects);//use objs to see the use of the NonMax Suppression algorithm
    }

    public ComputationGraph getNETWORK() {
        return NETWORK;
    }

}
