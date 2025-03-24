% Read and resize image
img = imread('dog.jpg');
img = imresize(img, [512 512]);

% Convert to HSV
hsvImg = rgb2hsv(img);

% Define refined HSV bounds for brown dog detection
lower_blue = [0.5, 0.2, 0.2];  
upper_blue = [0.7, 1.0, 1.0];

% Create HSV mask for non-blue (brown) objects
blueMask = (hsvImg(:,:,1) >= lower_blue(1) & hsvImg(:,:,1) <= upper_blue(1)) & ...
           (hsvImg(:,:,2) >= lower_blue(2) & hsvImg(:,:,2) <= upper_blue(2)) & ...
           (hsvImg(:,:,3) >= lower_blue(3) & hsvImg(:,:,3) <= upper_blue(3));

dogmask = ~blueMask;  

% Morphological operations to clean up the mask
se = strel('disk', 7);
dogmask = imclose(dogmask, se);
dogmask = imfill(dogmask, 'holes');
dogmask = bwareaopen(dogmask, 500);

% Segment the object
segmentedObject = bsxfun(@times, img, cast(dogmask, 'like', img));

% Object Detection: Bounding Box & Centroids
labeledMask = bwlabel(dogmask);
props = regionprops(labeledMask, 'BoundingBox', 'Centroid');

% K-Means Clustering on the Dog Region
labImg = rgb2lab(im2double(img));
[m, n, ~] = size(labImg);

% Extract only dog region pixels
maskedPixels = reshape(labImg, [], 3);
maskedPixels = maskedPixels(dogmask(:), :);

% Apply K-Means Clustering (3 Clusters)
k = 3; 
[cluster_idx, ~] = kmeans(maskedPixels, k, 'Distance', 'sqEuclidean', 'Replicates', 3);

% Convert Clusters to Image Form
segmentedLabels = zeros(m * n, 1);
segmentedLabels(dogmask(:)) = cluster_idx;
segmentedLabels = reshape(segmentedLabels, m, n);

% Overlay K-Means Clustering on Original Image
clusterColors = [150, 0, 150];  
coloredSegmentation = zeros(m, n, 3, 'uint8');

for j = 1:3
    coloredSegmentation(:,:,j) = uint8(segmentedLabels > 0) * clusterColors(j);
end

alpha = 0.5;
overlayedImg = uint8(double(img) * (1 - alpha) + double(coloredSegmentation) * alpha);

% Display Everything in a Single Figure
figure;
tiledlayout(2,3); % 2 Rows, 3 Columns for better visualization

% Binary Mask
nexttile;
imshow(dogmask);
title('Binary Mask');

% Segmented Dog
nexttile;
imshow(segmentedObject);
title('Segmented');

% Bounding Box & Centroid
nexttile;
imshow(img);
hold on;
for k = 1:length(props)
    rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 3);
    plot(props(k).Centroid(1), props(k).Centroid(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    text(props(k).Centroid(1), props(k).Centroid(2) - 10, ...
        sprintf('(%.1f, %.1f)', props(k).Centroid(1), props(k).Centroid(2)), ...
        'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');
end
hold off;
title('Bounding Box & Centroid');

% K-Means Clustering
nexttile;
imshow(segmentedLabels, []);
colormap jet;
colorbar;
title('K-Means');

% Overlayed K-Means Image
nexttile;
imshow(overlayedImg);
title('Overlayed K-Means');

