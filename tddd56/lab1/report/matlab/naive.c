parameters->mandelbrot_color.red=randint(255);
parameters->mandelbrot_color.green=randint(255);
parameters->mandelbrot_color.blue=randint(255);
parameters->begin_h = (args->id)*((parameters->height)/(NB_THREADS));
if (args->id!=(NB_THREADS-1)){
	parameters->end_h = ((args->id)+1)*((parameters->height)/(NB_THREADS));
}
else{
	parameters->end_h=parameters->height;
}
	parameters->begin_w = 0;
	parameters->end_w = parameters->width;
	compute_chunk(parameters);

