CPPFLAGS = -I/nethome/pkippes3/projects/cse6230/project2/prefix-gnucuda/include
OPENCV_LIB = /nethome/pkippes3/projects/cse6230/project2/prefix-gnucuda/lib
CUDA_LIB = /opt/cuda-4.0/cuda/lib64/
LDFLAGS =

#Libs: -L${libdir} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

#g++ -o $@ $(CPPFLAGS) -L$(LDFLAGS) $(OPENCV_LIB)/libopencv_core.a $^
test: test.cpp
	g++ -o $@ $(CPPFLAGS) -L$(OPENCV_LIB) $^ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann -lpthread -lrt -ltbb -Wl,-rpath=$(OPENCV_LIB)

clean:
	rm test
