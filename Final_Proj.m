imgs = {'f1.jpg', 'f1_2.jpg','luka.jpg','tiger.jpg'}; % Image file names
imgCount = length(imgs);
imgRes = cell(1, imgCount); % Preallocate cell array

% Read and resize images
for i = 1:imgCount
    img = imread(imgs{i});
    img = imresize(img, [512 512]); % Resize to 512x512
    imgRes{i} = img; % Store resized images
end

% Process each image
for i = 1:imgCount
    img = imgRes{i}; 
    imgHSV = rgb2hsv(img);
    hue = imgHSV(:,:,1); % Extract Hue channel
    
    % Define color masks
    redMask    = (hue > 0.9 | hue < 0.1);
    greenMask  = (hue > 0.25 & hue < 0.40);
    blueMask   = (hue > 0.55 & hue < 0.75);
    yellowMask = (hue > 0.12 & hue < 0.18);
    cyanMask   = (hue > 0.50 & hue < 0.55);
    magentaMask= (hue > 0.8 & hue < 0.9);
    
    
    segmentedImg = zeros(size(img), 'uint8');
    
    
    segmentedImg(:,:,1) = uint8(redMask) * 255 + uint8(magentaMask) * 255; 
    segmentedImg(:,:,2) = uint8(greenMask) * 255 + uint8(yellowMask) * 255; 
    segmentedImg(:,:,3) = uint8(blueMask) * 255 + uint8(cyanMask) * 255; 
    
   
    alpha = 0.5; % Transparency factor 
    overlayImg = uint8((1 - alpha) * double(img) + alpha * double(segmentedImg));

    % Display original, segmented, and overlay images
    figure;
    subplot(1, 3, 1); imshow(img); title('Original Image');
    subplot(1, 3, 2); imshow(segmentedImg); title('Segmented Colors');
    subplot(1, 3, 3); imshow(overlayImg); title('Overlay on Original');
end

