# LinesOnTheRoad

The purpose of this project is to create a module for automated driving.

The first step of automatic driving is to detect the line wehre the car is running nand keep it this way.


## Activities.

1. The lines are white and sometimes yellow (when there are provisional). So filtering of this colors will help the system to get the lines.
2. The lines are not going to the end of the horizont, so there is a ROI in front of our car that has to be used and the rest filtered out.
3. Then when the image left we shuld able to reduce the noise.
	- Blur
	- Canny.
	- Identify the lines
	- Filter some lines out.
	- Draw our own lines
	- 

