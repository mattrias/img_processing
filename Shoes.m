% Read and resize image
img = imread('shoes.jpg');
img = imresize(img, [512 512]);

% Convert to HSV
hsvImg = rgb2hsv(img);

% Define refined HSV bounds for green shoe detection
lower_hsv = [0.16, 0.3, 0.2];  % Adjust these bounds as needed
upper_hsv = [0.5, 1.0, 1.0];

% Create HSV mask
hsvMask = (hsvImg(:,:,1) >= lower_hsv(1) & hsvImg(:,:,1) <= upper_hsv(1)) & ...
          (hsvImg(:,:,2) >= lower_hsv(2) & hsvImg(:,:,2) <= upper_hsv(2)) & ...
          (hsvImg(:,:,3) >= lower_hsv(3) & hsvImg(:,:,3) <= upper_hsv(3));

% Morphological operations to clean up the mask
se = strel('disk', 7);
hsvMask = imclose(hsvMask, se);
hsvMask = imfill(hsvMask, 'holes');

% Segment the object
segmentedObject = bsxfun(@times, img, cast(hsvMask, 'like', img));

% Object Detection: Bounding Box & Centroids
labeledMask = bwlabel(hsvMask);
props = regionprops(labeledMask, 'Centroid', 'BoundingBox');

% Figure 1: Bounding Box and Centroid Visualization
figure;
imshow(img);
hold on;
for k = 1:length(props)
    rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);
    plot(props(k).Centroid(1), props(k).Centroid(2), 'ro');
    text(props(k).Centroid(1), props(k).Centroid(2), ...
        sprintf('(%0.1f, %0.1f)', props(k).Centroid(1), props(k).Centroid(2)), ...
        'Color', 'yellow', 'FontSize', 12);
end
hold off;
title('Detected Objects with Bounding Boxes and Centroids');

% Figure 2: Edge Detection within Bounding Boxes
figure;
for k = 1:length(props)
    bbox = round(props(k).BoundingBox);
    subImg = img(bbox(2):(bbox(2)+bbox(4)-1), bbox(1):(bbox(1)+bbox(3)-1), :);
    graySubImg = rgb2gray(subImg);
    edges = edge(graySubImg, 'canny');

    subplot(1, length(props), k);
    imshow(edges);
    title(['Edges in Box ', num2str(k)]);
end
title('Edge Detection within Bounding Boxes');

labImg = rgb2lab(im2double(img));
[m, n, ~] = size(labImg);

maskedPixels = reshape(labImg, [], 3);
maskedPixels = maskedPixels(hsvMask(:), :);

k = 3; 
[cluster_idx, ~] = kmeans(maskedPixels, k, 'Distance', 'sqEuclidean', 'Replicates', 3);

segmentedLabels = zeros(m * n, 1);
segmentedLabels(hsvMask(:)) = cluster_idx;
segmentedLabels = reshape(segmentedLabels, m, n);


figure;
for i = 1:k
    clusterMask = segmentedLabels == i;
    clusterImg = bsxfun(@times, img, cast(clusterMask, 'like', img));
    subplot(2, 2, i);
    imshow(clusterImg);
    title(['Cluster ', num2str(i)]);
end


clusterColors = [150, 0, 150];


coloredSegmentation = zeros(m, n, 3, 'uint8');
for j = 1:3
    coloredSegmentation(:,:,j) = uint8(segmentedLabels > 0) * clusterColors(j);
end

alpha = 0.5;
overlayedImg = uint8(double(img) * (1 - alpha) + double(coloredSegmentation) * alpha);


figure;
imshow(overlayedImg);
title('Overlayed K-Means Segmented Image on Original');
