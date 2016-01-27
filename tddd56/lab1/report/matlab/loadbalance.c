parameters->mandelbrot_color.red=randint(255);
parameters->mandelbrot_color.green=randint(255);
parameters->mandelbrot_color.blue=randint(255);
while(1){
	int myblocknumber;
	pthread_mutex_lock(&block_lock);
	myblocknumber = block_counter;
	if (myblocknumber!=100){
		block_counter+=1;
		pthread_mutex_unlock(&block_lock);
	}
	else{
		pthread_mutex_unlock(&block_lock);
		break;
	}
	parameters->begin_w=(myblocknumber%10)*parameters->width/10;
	if (myblocknumber%10!=9){
		parameters->end_w=(myblocknumber%10+1)*parameters->width/10;
	}
	else{
		parameters->end_w=parameters->width;
	}
	parameters->begin_h=(myblocknumber/10)*(parameters->height/10);
	if(myblocknumber/10+1!=10){
		parameters->end_h=(myblocknumber/10+1)*(parameters->height/10);
	}
	else{
		parameters->end_h=parameters->height;
	}
	compute_chunk(parameters);
}
