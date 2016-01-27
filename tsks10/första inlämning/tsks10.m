%Right click on signal-marso329.wav and choose import data. A window will
%appear and click on import in the upper right corner

%some constants
number_of_samples=length(data);
sampling_freqency=fs;

%plot freqency spectrum
X_mags = abs(fftshift(fft(data)));
bin_vals = 0 : number_of_samples-1;
N_2 = ceil(number_of_samples/2);
fax_Hz = (bin_vals-N_2)*sampling_freqency/number_of_samples;
figure(10);
plot(fax_Hz, X_mags)
xlabel('Frekvens [Hz]')
axis tight

%peaks at 39200(0,70000), 95000(70000,115000),133000(115000,20000)

%we have 7800000 samples and at samplefreq 400000 we have 19.5s of data
time_axis = 0:19.5/number_of_samples:19.5;
time_axis=time_axis(1:7800000);

% First peak through lowpassfilter
[B,A] = butter(10,0.35,'low');
first_peak_filtered = filter(B,A,data);
figure(2);
plot(time_axis,first_peak_filtered);
xlabel('Tid [s]');

% Second peak through bandpassfilter
[B,A] = butter(10,[0.35, 0.60],'bandpass');
second_peak_filtered = filter(B,A,data);
figure(3);
plot(time_axis,second_peak_filtered);
xlabel('Tid [s]');

% Third peak through highpassfilter
[B,A] = butter(10,0.6,'high');
third_peak_filtered = filter(B,A,data);
figure(4);
plot(time_axis,third_peak_filtered);
xlabel('Tid [s]');

%carrier freq is 133k Hz
carrier_freqency=133000;

%cross_correlation in white noise 
first_peak_cross_correlation = xcorr(first_peak_filtered,first_peak_filtered);
time_axis_cross = 0:39/15599999:39;
time_axis_cross=time_axis_cross(1:end-1);
figure(5);
plot(time_axis_cross,first_peak_cross_correlation);
xlabel('tau [s]');
axis([18.5,20.5,-200,250]);

%largest value at tau=19.5 and sidetops +-0.41
%0.41*400000=164000 samples
echo_samples=164000;
filtered_data = zeros(size(third_peak_filtered));
filtered_data(1:echo_samples) = third_peak_filtered(1:echo_samples);

%remove echo
for i = 0 : 45
temp1 = filtered_data((1+echo_samples*i):(echo_samples + echo_samples*i));
temp2 = third_peak_filtered((echo_samples+1+echo_samples*i):(i+2)*echo_samples);
filtered_data((echo_samples+1+echo_samples*i):(i+2)*echo_samples) = temp2 - 0.9*temp1;
end

%demodulation
I=((2*cos(2*pi*carrier_freqency*time_axis))').*filtered_data;
Q=((-2*sin(2*pi*carrier_freqency*time_axis))').*filtered_data;

% filter I and Q on the hearable spectrum
[B,A] = butter(10,0.10,'low');
i_filtered = filter(B,A,I);
q_filtered = filter(B,A,Q);

%downsample
q_audio=decimate(q_filtered, 4);
i_audio=decimate(i_filtered, 4);

%Uncomment which one you want to listen to
%soundsc(q_audio,100000)
soundsc(i_audio,100000)
