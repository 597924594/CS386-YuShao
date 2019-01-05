function  batch_ssim(total_num)
fid = fopen('fsim.txt', 'w');
% Table Header
fprintf(fid, 'Img_Seq     rgb_fsim      sketch_fsim\n\n');
fsim_sketch = 0;
fsim_photo = 0;
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
    fsim_photo = fsim(rgb0,rgb1);
    fsim_sketch = fsim(sketch0,sketch1);
    sum_photo = sum_photo + fsim_photo;
    sum_sketch = sum_sketch + fsim_sketch;
    y1 = [i;fsim_photo;fsim_sketch];
    fprintf(fid, '%1d    %f    %f\n', y1);
end
fprintf(fid, '\n\n average_rgb_fsim      average_sketch_fsim\n\n');
total_num=total_num-60;
y = [sum_photo/total_num;sum_sketch/total_num];
fprintf(fid, '%f    %f\n', y);
fclose(fid);
type fsim.txt