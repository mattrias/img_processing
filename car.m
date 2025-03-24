img = imread('car.jpg');
img = imresize(img, [512 512]);
hsvImg = rgb2hsv(img);

lower_red = [0.0, 0.5, 0.2];  
upper_red = [0.1, 1.0, 1.0];

redMask = (hsvImg(:,:,1) >= lower_red(1) & hsvImg(:,:,1) <= upper_red(1)) & ...
          (hsvImg(:,:,2) >= lower_red(2) & hsvImg(:,:,2) <= upper_red(2)) & ...
          (hsvImg(:,:,3) >= lower_red(3) & hsvImg(:,:,3) <= upper_red(3));

carMask = redMask;
se = strel('disk', 7);
carMask = imclose(carMask, se);
carMask = imfill(carMask, 'holes');
carMask = bwareaopen(carMask, 500);

segmentedObject = bsxfun(@times, img, cast(carMask, 'like', img));

labeledMask = bwlabel(carMask);
props = regionprops(labeledMask, 'BoundingBox', 'Centroid');

labImg = rgb2lab(im2double(img));
[m, n, ~] = size(labImg);
maskedPixels = reshape(labImg, [], 3);
maskedPixels = maskedPixels(carMask(:), :);

k = 3; 
[cluster_idx, ~] = kmeans(maskedPixels, k, 'Distance', 'sqEuclidean', 'Replicates', 3);

segmentedLabels = zeros(m * n, 1);
segmentedLabels(carMask(:)) = cluster_idx;
segmentedLabels = reshape(segmentedLabels, m, n);

clusterColors = [150, 0, 150];  
coloredSegmentation = zeros(m, n, 3, 'uint8');
for j = 1:3
    coloredSegmentation(:,:,j) = uint8(segmentedLabels > 0) * clusterColors(j);
end

alpha = 0.5;
overlayedImg = uint8(double(img) * (1 - alpha) + double(coloredSegmentation) * alpha);

figure;
tiledlayout(2,3);

nexttile;
imshow(carMask);
title('Binary Mask');

nexttile;
imshow(segmentedObject);
title('Segmented');

nexttile;
imshow(img);
hold on;
for k = 1:length(props)
    rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);
    plot(props(k).Centroid(1), props(k).Centroid(2), 'ro');
    text(props(k).Centroid(1), props(k).Centroid(2), ...
        sprintf('(%.1f, %.1f)', props(k).Centroid(1), props(k).Centroid(2)), ...
        'Color', 'yellow', 'FontSize', 12);
end
hold off;
title('Bounding Box & Centroid');

nexttile;
imshow(coloredSegmentation);
title('K-Means');

nexttile;
imshow(overlayedImg);
title('Overlayed K-Means');
