package com.emaraic.rubikcubedetector;

import com.emaraic.utils.YoloModel;
import java.util.concurrent.atomic.AtomicReference;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import org.bytedeco.javacpp.opencv_videoio;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_WIDTH;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.LoggerFactory;

/**
 * Created on Jul 4, 2018 , 2:37:28 PM
 *
 * @author Taha Emara 
 * Email : taha@emaraic.com 
 * Website: http://www.emaraic.com
 */
public class RubixDetector {

    private static final org.slf4j.Logger log = LoggerFactory.getLogger(RubixDetector.class);
    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private static final int IMAGE_INPUT_W = 416;
    private static final int IMAGE_INPUT_H = 416;

    public static void main(String[] args) {

        YoloModel detector = new YoloModel();

        final AtomicReference<opencv_videoio.VideoCapture> capture = new AtomicReference<>(new opencv_videoio.VideoCapture());
        capture.get().set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        capture.get().set(CV_CAP_PROP_FRAME_HEIGHT, 720);

        if (!capture.get().open(0)) {
            log.error("Can not open the cam !!!");
        }

        opencv_core.Mat colorimg = new opencv_core.Mat();

        CanvasFrame mainframe = new CanvasFrame("Real-time Rubik's Cube Detector - Emaraic", CanvasFrame.getDefaultGamma() / 2.2);
        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(600, 600);
        mainframe.setLocationRelativeTo(null);
        mainframe.setVisible(true);

        while (true) {
            while (capture.get().read(colorimg) && mainframe.isVisible()) {
                long st = System.currentTimeMillis();
                opencv_imgproc.resize(colorimg, colorimg, new opencv_core.Size(IMAGE_INPUT_W, IMAGE_INPUT_H));
                detector.detectRubixCube(colorimg, .4);
                double per = (System.currentTimeMillis() - st) / 1000.0;
                log.info("It takes " + per + "Seconds to make detection");
                putText(colorimg, "Detection Time : " + per + " ms", new opencv_core.Point(10, 25), 2,.9, opencv_core.Scalar.YELLOW);

                mainframe.showImage(converter.convert(colorimg));
                try {
                    Thread.sleep(20);
                } catch (InterruptedException ex) {
                    log.error(ex.getMessage());
                }
            }
        }
    }
}
