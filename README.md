# NTSC_Analog_Television
 A script that converts an image into an NTSC signal and creates an animation of a CRT raster scan of that image.

This was originally for a class project in which I had to choose an image system to research and demonstrate. Analog TV, especially the technology behind color TV, was an interesting intersection of RF, image processing, and the human visual system.

Some creative liberties were taken to better illustrate the physical process of CRT scans. The three cathode rays are, in reality, all just streams of electrons (not distinct colors as shown), but coloring these beams in the animation helps demonstrate the effect of the active color components on the resulting image. From what I gather, many or all CRTs begin their scan in the top middle of the screen, causing a sort of half-line; this was omitted in favor of starting at the screen's corner. The amount of phosphor dots is less than the typical number in CRT screens, but I found this to be a good mid-point between a small number of larger dots that more clearly show the effect of the phosphor being energized, and the more realistic large number with smaller dots which creates a smooth image (but would slow down the animation).

 The animation is done with successive frames of a MATLAB figure, which, admittedly, is slow. I might optimize the speed in the future, or port the code to other platforms.
