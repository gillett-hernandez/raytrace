# raytrace
Python raytracer

The speed difference is explained at http://www.excamera.com/sphinx/article-ray.html

additions compared to the original:
multiprocessing
pygame integration for live viewing and camera movement
camera rotation and movement (in rotation branch)
multisampling (multiple rays per pixel) (in sampled-raytrace branch)
refractive materials and tracing (in glass branch)

todo:
caustics and better shadows
lights with a defined radius (to allow for soft shadows)
other shader types?
maybe path tracing instead of ray tracing to address some of the above todos?
