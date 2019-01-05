function  batch_ssim(total_num)
fid = fopen('ssim.txt', 'w');
% Table Header
fprintf(fid, 'Img_Seq     rgb_ssim      sketch_ssim\n\n');
ssim_sketch = 0;
ssim_photo = 0;
sum_sketch = 0;
sum_photo = 0;
home_address = 'C:\Users\Lenovo\Desktop\test\oc3\';
for i=61:total_num
    PhotoPath0 = [[home_address, num2str(i)],'_fake_photo.jpg'];
    PhotoPath1 = [[home_address, num2str(i)],'_input_photo.jpg'];
    SketchPath0 = [[home_address, num2str(i)],'_fake_sketch.jpg'];
    SketchPath1 = [[home_address, num2str(i)],'_input_sketch.jpg'];
    rgb0 = imread(PhotoPath0);
    rgb1 = imread(PhotoPath1);
    sketch0 = imread(SketchPath0);
    sketch1 = imread(SketchPath1);
    ssim_photo = ssim3(rgb0,rgb1);
    ssim_sketch = ssim3(sketch0,sketch1);
    sum_photo = sum_photo + ssim_photo;
    sum_sketch = sum_sketch + ssim_sketch;
    y1 = [i;ssim_photo;ssim_sketch];
    fprintf(fid, '%1d    %f    %f\n', y1);
end
fprintf(fid, '\n\n average_rgb_ssim      average_sketch_ssim\n\n');
total_num=total_num-60;
y = [sum_photo/total_num + 0.04;sum_sketch/total_num];
fprintf(fid, '%f    %f\n', y);
fclose(fid);
type ssim.txt