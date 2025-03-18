imgs = {'paprika.jpg'}; % General image file names
imgCount = length(imgs);
imgRes = cell(1, imgCount); % Preallocate cell array

img = imread('paprika.jpg');
img = imresize(img, [512 512]);
imgRes{1} = img; 

rgbImg = img;
hsvImg = rgb2hsv(img);
ycbcrImg = rgb2ycbcr(img);

lower_bg = [100, 70, 40];   
upper_bg = [210, 200, 200]; 

backgroundMask = (img(:,:,1) >= lower_bg(1) & img(:,:,1) <= upper_bg(1)) & ...
                 (img(:,:,2) >= lower_bg(2) & img(:,:,2) <= upper_bg(2)) & ...
                 (img(:,:,3) >= lower_bg(3) & img(:,:,3) <= upper_bg(3));

objectMask = ~backgroundMask;
se = strel('disk', 5); 
objectMask = imopen(objectMask, se);  
objectMask = imclose(objectMask, se); 
objectMask = imfill(objectMask, 'holes');

% Segment the objects using the mask
segmentedObjects = bsxfun(@times, img, cast(objectMask, 'like', img));

% Display segmented image
figure;
imshow(segmentedObjects);
title('Segmented Objects (Background Removed)');

% Thresholds in HSV for refined object detection (adjust as necessary)
hueThresh = (hsvImg(:,:,1) < 0.1) | (hsvImg(:,:,1) > 0.9); % Exclude specific hues
satThresh = hsvImg(:,:,2) > 0.3; % Saturation threshold
valThresh = hsvImg(:,:,3) > 0.2; % Brightness threshold

mask = satThresh & valThresh & ~hueThresh;


mask = bwareaopen(mask, 500); % Remove small objects
mask = imfill(mask, 'holes'); % Fill holes

% Label connected regions
labeledMask = bwlabel(mask);

% Extract properties: Area, Centroid, and Bounding Box
props = regionprops(labeledMask, 'Area', 'Centroid', 'BoundingBox');

% Display the original image with bounding boxes and centroids
figure;
imshow(img);
hold on;

for k = 1:length(props)
    % Draw bounding box around each object
    rectangle('Position', props(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 2);

    % Mark centroid
    plot(props(k).Centroid(1), props(k).Centroid(2), 'ro');

    % Display area as a label
    text(props(k).Centroid(1), props(k).Centroid(2), ...
        sprintf('%d', props(k).Area), 'Color', 'yellow', 'FontSize', 12);
end

hold off;
title('Detected Objects with Bounding Boxes and Centroids');

%Edge Detection
grayImg = rgb2gray(img);

edges = edge(grayImg, 'canny');
se = strel('disk', 2);
dilatedEdges = imdilate(edges, se);

figure;
imshow(edges);
imshow(dilatedEdges);

%K-Means
img = im2double(img); 

labImg = rgb2lab(img);
[m, n, c] = size(labImg);
maskedPixelData = reshape(labImg, [], 3);
maskedPixelData = maskedPixelData(objectMask(:), :);
k = 3; 

[cluster_idx, cluster_center] = kmeans(maskedPixelData, k, 'distance', 'sqEuclidean', ...
                                       'Replicates', 3);

segmentedLabels = zeros(m * n, 1);
segmentedLabels(objectMask(:)) = cluster_idx;
segmentedLabels = reshape(segmentedLabels, m, n);

cleanedSegments = zeros(m, n, 3);

figure;
for i = 1:k
    clusterMask = segmentedLabels == i;
    
    clusterMask = bwareaopen(clusterMask, 500); 
    
    clusterSegment = bsxfun(@times, img, cast(clusterMask, 'like', img));
    
    cleanedSegments = cleanedSegments + clusterSegment;
        
    subplot(2,2,i);
    imshow(clusterSegment);
    title(['Cleaned Cluster ', num2str(i)]);

end

subplot(2,2,4)
imshow(cleanedSegments);
title('Final Cleaned K-means Segmentation');

overlayColor = [0, 190, 200] / 255; % Normalize to [0,1] range

% Initialize the overlay image
coloredOverlay = zeros(m, n, 3);

% Iterate through all clusters and apply the overlay color
for i = 1:k
    clusterMask = segmentedLabels == i; % Mask for cluster i
    
    for j = 1:3
        coloredOverlay(:,:,j) = coloredOverlay(:,:,j) + clusterMask * overlayColor(j);
    end
end

% Blend with the original image
alpha = 0.3; % Adjust transparency (lower is more transparent)
overlayedImage = img .* (1 - alpha) + coloredOverlay * alpha;

% Display overlayed image
figure;
imshow(overlayedImage);
title('K-Means Segmented Overlay (Colored & Transparent)');


