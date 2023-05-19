clear all; clf;

originalDir = dir;
srcdir = uigetdir('',...
    'Choose a folder containing the image(s) you want to turn into a CRT television scan.');

stills = [];
for exten=["png", "jpg", "jpeg"]
    extrfiles = dir(fullfile(srcdir,strcat('*.',exten)));
    stills = [stills fullfile(srcdir,string({extrfiles.name}))];
end

%                    Enter your parameters below, if desired
%
% first is an array of strings of the image files you want as frames
% second is frame rate of resulting animation
% third is if you want an animation saved at all
% fourth is if you want to shorten the vertical sync to artificial duration

movieManager(stills, 15, 1, 0);
% movieManager([uigetfile()], 15, 1, 1);


% function definitions below.
% movieManager controls most of the sequence. Before looping through
% everything it calls a few functions to create the different subplots.
% Then, for each frame of animation, it will create the NTSC signal by 
% calling ntscTx(), and start looping through that signal and updating the 
% graphs, creating essentially a slowed down view at a live NTSC CRT.

% ---------------------------------------------------------------------

function [a] = movieManager(frames, fps, saveAnimation, shortenVertSync)
% Call ntscTx to create the actual NTSC signals and 'cheat' values for YIQ
% (Skips any demodulation)

% Create the overall multi-plot figure on which things will be plotted
ntscfig = figure(1);
set(gcf,'PaperPositionMode','auto')
set(gcf,'InvertHardCopy','off')
pixelsx = 1280; pixelsy = 720;
xoffscreen = 20; yoffscreen = 60;
set(ntscfig, 'Position',[xoffscreen yoffscreen pixelsx pixelsy]);
% possibly other settings here for formatting the figure for export
set(gcf, 'Color',[0 0 0]);

start2stop = @(aa) aa(1):aa(2); % shorthand

% set up some values
u=10^(-6); % millisec
lt = 63.49*u;
resize = [480/2 640/2]; % will resize all input images to this.
margin =  30; % margin in pixels on each side of rendered CRT image
crtims = (margin*2)+resize;

% The "stripchart recorder" plot --------------------------------
% scsp = subplot(9, 9, [(9*7)+1:9*8 (9*8)+1:9*9]);
mgn = 0.05;
scsp = axes('Parent',gcf(),'Units','normalized',...
            'Position',[0+mgn 0+(1.5*mgn) 0.5-(2*mgn) 0.4]);
makeStripChart(scsp);
scln = plot([0],[0],'w');
ylim([-0.1 1.1]);
xlim([0 lt*2]);

% The IQ plot --------------------------------
% iqsp = subplot(9, 9, [1:3 10:12 19:21]);
iqsp = axes('Parent',gcf(),'Units','normalized','Position',[-0.08 0.55 0.43 0.4]);
makeIQ(iqsp,1,1);
axes(iqsp);
% Establish some data lines that will be updated in the loop.

iqln = plot([-0.01 0.01],[-0.01 0.01],'-',...
    'Color',[1 1 1 0.3],'LineWidth',2);

% Color blob --------------------------------
% iqsp = subplot(9, 9, [1:3 10:12 19:21]);
cbsp = axes('Parent',gcf(),'Units','normalized','Position',[0.2 0.88 0.1 0.1]);
axes(cbsp);
hold on; axis equal;
grid off; axis off;
radi = 0.5;
cent = [0 0];
cbln = rectangle('Position',[cent-radi 2*radi 2*radi],'Curvature',[1 1],...
    'FaceColor','k','EdgeColor',[0.6 0.6 0.6]);
title('Current Color');


% The CRT screen --------------------------------
% crtsp = subplot(9, 9, [5:9 14:18 23:27 31:36 40:45]) % 5/9 x 5/9
% crtsp = subplot(9, 9, [4:9 13:18 22:27 30:36 39:45 48:54]); % 6/9 x 6/9
crtsp = axes('Parent',gcf(),'Units','normalized','Position',[0.26 0.2 0.7 1]);
hold on;
axes(crtsp);
set(crtsp, {'YDir'}, {'reverse'}); % reverse, following down-y right-x mode
set(crtsp, 'color','none');
axis off; axis equal;

% Make the CRT --------------------------------------------
% image size (assumes all frames of the sequence will be same size...
%TODO: consider doing imresize(imread(frames(f)),[480 640])
imgsz = resize; %size(imread(frames(1)));
imsx = imgsz(2); imsy = imgsz(1); %image size dimensions

% The function that actually generates the CRT
% using a render pixel size 4x larger than image pixel size)
fprintf('Creating the CRT now...\n')
tic % [crtaxindex, crtim, rgbcrt, alphacrt, rays]
renderScale = 8; % rendered CRT will be this many times larger than image
offset = 160; % offset/padding on outside of rendered CRT screen, in pixels
[crtd, crtimage, crtrgb, crtalpha, rays] = makeCRT(crtsp, imsx*renderScale, imsy*renderScale, offset);
alphabase = crtalpha; % this will be the minimum/base alpha matrix
rgbmask = double(crtrgb>0); % to be used as multiplier later
rgbsize = size(crtalpha);
rgblen = rgbsize(1); rgbwid = rgbsize(2);
toc

% Set up timing stuff, organize key strings etc -----------------------------------
% Sample time of original signal
ts = 1/(13.5*10^9); % must be pretty small to work well with ntscTx

% Samples of the downsampled signal
dss = 10; % downsample by this many steps
dst = ts*dss; % downsample time

% Samples per frame of animation.
giffps = 30; % frames per second of gif or other animation
% However, we don't want to go 1:1 because you wouldn't see any crt action
slowedby = 100000; % animated sequence is this many times slower than what it would be in real life.
% So, our frequency of viewing the info is basically giffps*slowedby.
viewfreq = giffps*slowedby;
viewtime = 1/viewfreq;
viewsamps = round(viewtime/dst); % view time over downsample time:  # samples in a view

pgro = 0.8; % phosphor growth amount per sample (scaled down further more later)
phospread = 0.8; % radius of phosphor spread (relative to pixel size of image and phosphors
degspread = 3; % degree of the shape of the spread

sclent = lt; % stripchart length (time)
iqlent = lt/20; % iq length (time)

% pfields = fieldnames(phos);

rgbs=["r" "g" "b"]; % RGB strings
srgb.r=1; srgb.g=2; srgb.b=3;

% Create a folder in which the images will be stored:
if saveAnimation
    animfoldername = "NTSC_Animation_"+regexprep(datestr(datetime),["-",":"," "],["_","_","_"]);
    fprintf('Saving animation files to the folder %s\n...',animfoldername);
    mkdir(animfoldername);
end

% The main loop -----------------------------------------------

oic = 1; % output image count
for f=1:length(frames)
    img = imread(frames(f));
    % if plan on irregular images, consider doing 
    img = imresize(img,resize);

    % First call the function that will make the signal from the image frame.
    fprintf('Pre-rendering the NTSC signal for frame %d of input sequence (could take a minute)...\n', f);
    tic
    [nt, y, i, q, dx dy] = ntscTx(img,ts,0,0,imsx,imsy,shortenVertSync);
    nt=nt(1:dss:end); % downsample
    y=y(1:dss:end); % downsample
    i=i(1:dss:end); % downsample
    q=q(1:dss:end); % downsample
    dx=round(offset)+(renderScale*dx(1:dss:end)); % downsample
    dy=round(offset)+(renderScale*dy(1:dss:end)); % downsample
    toc
    sl = length(nt(1,:)); % signal length in num samples
    pdec = 0.9/(2*(sl/viewsamps)); % phosphor decay amount (subtractive)

    decay = 0.99999; % multiplicative decay %TODO: make dependent on frame rate, etc
    alpmax = 0.9; % alpha maximum; display will not go higher than this

    rgb = ntsc2rgb([y; i; q]');
    % More timing
    iqlens = iqlent/dst; % number of samples to keep on plot at once
    sclens = sclent/dst; 
    % for still showing colorburst as dotted line:
    sampsforburst=round((0.56*u)/dst);
    % Downsample
    % take the floor of the col/ts
    % will have % across
    % colInd = floor(percentAcross*numCols)
    % rowInd = 
    
    % for rays' end positions. mostly copied from crt function 
    lineHeight = (3*imsy)/((3*imsy) + 4); % this shouldn't... but only works for even number of lines?
    r = lineHeight*2/3;
    apo = r*sqrt(3)/2;
    colWidth = apo*6;
    cox = [-colWidth/6 colWidth/6 0];  % color offset x
    coy = [-apo/2 -apo/2 r]; % color offset y

    % create beam kernel
    beamn = round(offset/2)-mod(round(offset/2),2)+1;
    beamo = (beamn-1)/2; % offset from beam; depends only on beamn value
    % create beam matrix:
    beam = gaussianKernel(beamn, 10, 0);
    beamtemp = zeros(size(beam));
    beamold = zeros(size(beam));
    % scale beam matrix from 1 to 1.9 (will bee multiplied with current
    % values to obtain new values):
    beam = (beam*0.5);
    % matrix to be filled in each animation frame with 
    % indices within crtrgb where beam matrix starts and ends, where:
    % row 1 x, row 2 y; col 1 min, col 2 max; page 1 red, p 2 grn, p 3 blue
    beamends = zeros(2,2,3); 
    meammask = zeros(size(crtalpha));


    % For each animation (gif etc) frame:
    fprintf('Looping through raster scan for frame %d now...\n',f);
    for af = 1:viewsamps:sl
%         tic
        % update IQ
        iqln.XData = [iqln.XData i(af:af+viewsamps-1)];
        iqln.YData = [iqln.YData q(af:af+viewsamps-1)];
        % cut out the earlier parts of the data if we have filled up the
        % buffer
        if length(iqln.XData)>iqlens
            iqln.XData(1:viewsamps-1)=[];
            iqln.YData(1:viewsamps-1)=[];
        end
    
        % update stripchart
        scln.YData = [scln.YData nt(af:af+viewsamps-1)];
        scln.XData = [scln.XData dst*[af-1:af+viewsamps-2]./u];
        % cut out the earlier parts of the data if we have filled up the plot
        if length(scln.XData)>sclens
            scln.XData(1:viewsamps-1)=[];
            scln.YData(1:viewsamps-1)=[];
        end
        % update axis
        axes(scsp);
%         xlim tight;
        scxmin = min(scln.XData); % stripchart x min
        scxmax = max(scln.XData); % stripchart x min
        % update limits so chart width is constant over time:
        xlim([scxmin max(scxmax,scxmin+(sclent/u))]);
        
        % move rays. hexagonal/triangular offsets are flipped and added
        % (reflect across shadow mask essentially)
        rays.rray.XData(3) = dx(af)+cox(1); 
        rays.rray.YData(3) = dy(af)+coy(1);

        rays.gray.XData(3) = dx(af)+cox(2); 
        rays.gray.YData(3) = dy(af)+coy(2);

        rays.bray.XData(3) = dx(af)+cox(3); 
        rays.bray.YData(3) = dy(af)+coy(3);

        rays.cray.XData(3) = dx(af); 
        rays.cray.YData(3) = dy(af);
        
        % change intensity of rays.
        rays.rray.Color(4) = rgb(af,1);
        rays.gray.Color(4) = rgb(af,2);
        rays.bray.Color(4) = rgb(af,3);
        cbln.FaceColor = rgb(af,:);


% old method using polygons in a plot to represent the CRT screen
% (newer method updates a matrix that is plotted as an image,
% but there might be benefit in using the previous style in the future)

%         % Update phosphors:
%         % loop through all phosphor properties. If they are close enough to the
%         % incident ray, add some to the alpha value. If not, subtract some.
%         % phosphor properties were enumerated before loop.
%         %
%         % Calculate RGB intensities
%         % (only do if not retracing or at black level)
%         if ~all(~rgb(af,:)) % only if all aren't zero
%             for pii = 1:numel(pfields)
%                 phostr = pfields{pii};
%                 pcoord = regexp(phostr,'_','split');
%                 phoclr = pcoord{1};
%                 phoxpos = imsx*str2num(pcoord{2})/10000;
%                 phoypos = imsy*str2num(pcoord{3})/10000;
%                 % with this position information, determine if phosphors
%                 % intensity should increase or decrease:
%                 raypos = [dx(af), dy(af)]; % current ray position
%                 phos2raydist = sqrt((raypos(1)-phoxpos)^2 + (raypos(2)-phoypos)^2);
%                 % add or subtract
%                 if phos2raydist <= phospread
%                     % depending on color, apply different power
%                     intenscale = rgb(af,srgb.(phoclr)); % where gammas might come in
%                     pixcolscl = (intenscale*0.9);
%                     % previous code that fades the intensity over distance
%                     % from beam center, removed for simplicity:
%                     newphosintens = pixcolscl*pgro*((1-(phos2raydist/phospread))^degspread)/(phospread^degspread);
% %                     newphosintens =
% %                     pixcolscl*pgro*(1-(phos2raydist/phospread)); % simplified
%                     % this should prevent screen from "saturating" all colors.
%                     appliedcolor = min([pixcolscl+0.1, newphosintens+phos.(phostr).FaceAlpha]);
%                     % Now shouldn't decrease any values (wouldn't be realistic),
%                     % if anything keep constant.
%                     phos.(phostr).FaceAlpha = max([appliedcolor, 0.1, phos.(phostr).FaceAlpha]);
%                 else
%                     if phos.(phostr).FaceAlpha ~= 0.1
%                         % hoping this way, by not updating graph items, animation will go faster.
%                         phos.(phostr).FaceAlpha = max([phos.(phostr).FaceAlpha-pdec, 0.1]);
%                     end
%                 end
%                 
%                 
%             end
%         end
        

        % perform the standard decay for this frame of animation:
        crtalpha = max(decay*crtalpha,alphabase);

        % re-do the RGB and alpha matrices for this frame of animation
        %         % Calculate RGB intensities
        % (only do if not retracing or at black level)
        if ~all(~rgb(af,:)) % only if all aren't zero


            % calculate the kernel indices:
            % kernel was created earlier

            % convert the ray positions to indices of the crtrgb matrix:
            % format of beamends:
            % row 1 x, row 2 y; col 1 min, col 2 max; page 1 red, p 2 grn, p 3 blue
            beamends(1,:,1) = round(rays.rray.XData(3))+[-beamo,beamo]; 
            beamends(2,:,1) = round(rays.rray.YData(3))+[-beamo,beamo];
            
            beamends(1,:,2) = round(rays.gray.XData(3))+[-beamo,beamo]; 
            beamends(2,:,2) = round(rays.gray.YData(3))+[-beamo,beamo];
            
            beamends(1,:,3) = round(rays.bray.XData(3))+[-beamo,beamo]; 
            beamends(2,:,3) = round(rays.bray.YData(3))+[-beamo,beamo];


            % loop between red, green, and blue:

            for color = 1:3
                % the new values for this chunk of the crt alpha matrix are:
                % <the old alpha values in this chunk> * <rgb mask in this
                % chunk> * <beam matrix (same size as this chunk)>
                % not to subceed corresponding values in the alphabase, or
                % exceed value of alpmax

                %TODO: break this off into a separate function
                % check ray positon; something is not scaling correctly:
                % x4?

%                 beamtemp = min(...
%                     max(crtalpha(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color))) .* rgbmask(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)),color) .* beam,...
%                     alphabase(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)))),alpmax);
%                 % wherever the current color is not valid, replace the new
%                 % values with what we already had. First, save the old data
%                 % to an appropriately sized matrix:
%                 beamold = crtalpha(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)));
%                 % next, assign the nonvalid colors' indices' values to temp matrix:
%                 beamtemp(rgbmask(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)),color)==0) = beamold(rgbmask(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)),color)==0);
%                 % now assign temp matrix to the CRT alpha matrix.
%                 crtalpha(start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color))) = beamtemp;
                %
%                 beamends(1,2,color);
                crtalpha(start2stop(beamends(2,:,color)),...
                    start2stop(beamends(1,:,color))) = min(alpmax,...
                    crtalpha(start2stop(beamends(2,:,color)),...
                    start2stop(beamends(1,:,color))) +...
                    (rgb(af,color) * rgbmask(start2stop(beamends(2,:,color)),...
                    start2stop(beamends(1,:,color)),color).*beam));

                

                %
%                 valids = ismembc(reshape([1:rgblen*rgbwid],[rgblen,rgbwid]),...
%                     sub2ind(size(crtalpha),...
%                     start2stop(beamends(2,:,color)),...
%                     start2stop(beamends(1,:,color)))) & rgblogical(:,:,color);
% 


            end


        end


        % apply the new image to the CRT screen in the figure
        crtimage.AlphaData = crtalpha;
%         pause(0.2); % pause to allow plot to update

%         toc

        %making the animation files (individual frames first)
        if saveAnimation
            filename = sprintf('ntsc%012d',oic);
            saveas(ntscfig,animfoldername+"/"+filename+".png");
        end
        oic = oic+1;
    
    end
end

fprintf('End of supplied frames.');

% now take all the exported individual frames and turn into an MP4 video
if saveAnimation
%     animfoldername = uigetdir();
%     cd(animfoldername);
%     fps = 60;
    
    cd animfoldername;

    imgs = dir('*.png');
    numFrames = length(imgs);

    % gif version:
    f = imread(imgs(1).name); % load first frame as setup
    [f,map] = rgb2ind(f,256,'nodither');
    anim=repmat(f, [1 1 1 numFrames]);

    for k=1:numFrames
        f = imread(imgs(k).name);
        anim(:,:,1,k) = rgb2ind(f,map,'nodither');
    end
    % gif version:
    imwrite(anim, map, strcat(animfoldername, '.gif'),...
        'DelayTime', 0, 'LoopCount', inf);

%     % video file version:
%     vwriter = VideoWriter(animfoldername,'MPEG-4')
%     vwriter.FrameRate = fps;
%     open(vwriter);
%     for vid=1:numFrames
%         imgv = imgs(vid).name;
%         I = imread(imgv);
%         writeVideo(vwriter, I);
%     end
%     close(vwriter);

    pause(1);
    fprintf('Animation complete, saved as %s.\n', animfoldername)
end


end


%% ---------------

function [ntsc, ydat, idat, qdat, defx, defy] = ntscTx(im, ts, prog,...
    graysc, xsize, ysize, shortenVerticalSync)
% (Note that the coordinates used for defx and defy are relative to the
% scaled source image, not the rendered CRT image, which would be larger.

% if arguments are missing, make some assumptions:
if nargin<4
    graysc = 0; % not grayscale (color)
    if nargin<3
        prog = 0; % not progressive (interlaced)
    end
end

% Following 3 RF values not used as of v09
fpc = 61.25*10^6; % Channel 3
fco = 3.58*10^6; % color offset frequency
fcc = fpc+fco; % carrier frequency for color
% fs = 13.5*10^9; % Seems that 13.5 MHz is standard; Do 1000x that so things still ine up ok
% ts = 1/fs;
u=10^(-6);
% u=1;
% ts=0.0001;

% Condition the input image
%
% Convert to grayscale if needed
imsz = size(im);
if length(imsz)==3 && graysc % if given image is color but we want gray
    fprintf('Converting input image to grayscale...')
    im = rgb2gray(im);
elseif length(imsz)==2 && ~graysc % if want color but input is gray
    fprintf('Given image is not color; treating as grayscale.')
    graysc = 1;
end
%
% Convert to double
imout = double(im);
%
% 

if prog
    fprintf('Progressive scan not yet suppported in this simulation.')
    prog = 0;
end
fieldr = 59.94; % in Hz
framer = fieldr/2;
% lines = 525;
lines = imsz(1);
if shortenVerticalSync
    vrls = 1;
    fprintf('Artificially reducing vertical resync for illustrative purposes.\n');
else
    vrls = 21; % number of horizontal lines in a vertical retrace
end
% Voltage levels for sync/blank parts of signal (units of IRE):
blk = 0.3; % black level
flr = 0; % floor (blacker than black)
ysc = 0.7; % Y (Luminance) scale
crm = 0.2; % chroma modulating scale
brs = crm; % color burst amplitude

% horizontal sync/blanking timings in us:
timing.frontPorch = 1.5*u;
timing.lineSync = 4.7*u;
timing.backPorch = 4.7*u; % color burst rides on this for about 2.5 us
timing.actualVideo = 52.59*u;
timing.horizBlank = timing.frontPorch + timing.lineSync + timing.backPorch;
timing.oneLine = timing.horizBlank + timing.actualVideo;
% vertical sync/blanking timings in us:
% timing.invisibleVertLinesPerField = 19; % or 21? 23?
% timing.vertBlank = timing.oneLine * timing.invisibleVertLinesPerField;


% create non-image part of one line: horiz. blanking and syncing
lineBlank = [blk*ones(1,round(timing.frontPorch/ts))...
    flr*ones(1,round(timing.lineSync/ts))...
    blk*ones(1,round(timing.backPorch/ts))];
% times = [0:ts:ts*length(lineBlank)-ts]; % not necessary

% vertical -----------------------------------------------
% create non-image part of one line: horiz. blanking and syncing

equPulseLine = [blk*ones(1,round(timing.frontPorch/ts))...
    flr*ones(1,round(timing.lineSync/ts))...
    blk*ones(1,round(((25.49*u)+timing.backPorch)/ts))...
    flr*ones(1,round(timing.lineSync/ts))...
    blk*ones(1,round((22.4*u)/ts))];
if shortenVerticalSync % fake/short
    vBlankEven = equPulseLine;
else % the full series of blanking and syncing
    vertSyncLine = (flr+blk)-equPulseLine(end:-1:1);% flip, reverse, shift to represent sync pulse shapes
    emptyLine = [lineBlank blk*ones(1,round(timing.actualVideo/ts))];
    
    vBlankEven = [repmat(equPulseLine(1,:),[1 3])...
                  repmat(vertSyncLine(1,:),[1 3])...
                  repmat(equPulseLine(1,:),[1 3])...
                  repmat(emptyLine(1,:),[1 12])];
end



lineSams = round((timing.oneLine)/ts); % samples per line
blankSams = round((timing.horizBlank)/ts); % samples per blank period
actualVideoSams = round((timing.actualVideo)/ts);

% create the main signals that will be returned:
siglen = lineSams*(lines+(2*vrls));
ntsc = ones(1,siglen); % frame signal
ydat = zeros(1,siglen); % Luma
idat = zeros(1,siglen); % Chroma I
qdat = zeros(1,siglen); % Chroma Q
defx = zeros(1,siglen); % Deflection of rays in x (0-1)
defy = zeros(1,siglen); % Deflection of rays in y (0-1)


colorBurstDuration = 9/fco;

% create the image Y, I, and Q sequences
if ~graysc
    imYIQ = rgb2ntsc(im); 
    Y = imYIQ(:,:,1);
    I = imYIQ(:,:,2);
    Q = imYIQ(:,:,3);
else
    Y = im; % take the luminance values
end

% Interpolation
% Create fakeYtimes by scaling the indices to the duration of line data:
fakeYtimes = [0:length(Y(1,:))-1]*(timing.actualVideo-ts)/(length(Y)-1); 
% create time samples at which the image data will be interpolated, to
% fit sample rate of signal:
realYtimes = [0:ts:timing.actualVideo-ts];

il = ~prog; % Invert progressive scan, for looping purposes
l = 1; % line (signal order, sequence in interlaced stream, incl vert retrace)
for iil = 1:(1+il) % 1 to 2 or just 1
    for ln = iil:1+il:lines % every other or just progressive
        %
        istart = ((l-1)*lineSams)+1+blankSams;
        iend = l*lineSams;
        % assign the first part of this line as the blanking period
        bihb =((l-1)*lineSams)+1; eihb = ((l-1)*lineSams)+blankSams; %begin and end index for hor. blank
        ntsc(1,bihb:eihb) = lineBlank(1,:);
        blankLenH = length(lineBlank(1,:)); 
        % horizontal blanking/sync/retrace
        if ln==1 || (iil==2 && ln==2) % for the horiz retrace before first or second lines
            % don't move for first line's blanking 
            defy(1,bihb:eihb) = (ln-0.5)*ones(1,blankLenH);
            defx(1,bihb:eihb) = zeros(1,blankLenH);
        else
            % otherwise, typical retrace:
            defy(1,bihb:eihb) = linspace(ln-0.5,max(0.5,ln-2.5),blankLenH);
            defx(1,bihb:eihb) = linspace(xsize,0,blankLenH);
        end
        
        % assign the next part as the actual data
        realY = interp1(fakeYtimes, Y(ln,:), realYtimes, 'spline');
        % This is where the Luminance Y is actually applied:
        ntsc(1,istart:iend) = blk+(0.7*realY(1,:)); % could add Gamma correction somewhere here
        % x and y deflection of CRT rays
        actLenSams = length(ntsc(1,istart:iend)); %TODO: make more efficient
        defx(1,istart:iend) = linspace(0,xsize,actLenSams); % deflection x
        defy(1,istart:iend) = (ln-0.5)*ones(1,actLenSams);  % deflection y; constant for straight line
        % Now add the Chrominance (I and Q):
        if ~graysc
            % Add Chroma
            % First need to interpolate to get I and Q, the modulating
            % signals, to be sampled properly:
            realI = interp1(fakeYtimes, I(ln,:), realYtimes, 'spline');
            realQ = interp1(fakeYtimes, Q(ln,:), realYtimes, 'spline');
            QAM = crm*(realI.*cos(2*pi*fco*[(istart-1)*ts:ts:(iend-1)*ts]) + ...
                       realQ.*sin(2*pi*fco*[(istart-1)*ts:ts:(iend-1)*ts]));
            % Assign main signals for the active video part of this line
            ntsc(1,istart:iend) = ntsc(1,istart:iend) + QAM;
            ydat(1,istart:iend) = realY;
            idat(1,istart:iend) = realI;
            qdat(1,istart:iend) = realQ;

            %Modulate color burst onto each backporch. 
            startTimeBeforeActive = (timing.backPorch/2) +(colorBurstDuration/2);
            endTimeBeforeActive = (timing.backPorch/2) - (colorBurstDuration/2);

            iBurstStart = istart - floor(startTimeBeforeActive/ts);
            iBurstEnd = istart - floor(endTimeBeforeActive/ts);
            timesSinceFrameStart = (iBurstStart-1)*ts:ts:(iBurstEnd-1)*ts;
            ntsc(1,iBurstStart:iBurstEnd) = ntsc(1,iBurstStart:iBurstEnd) +...
                brs*sin(2*pi*fco*timesSinceFrameStart);
            burstStart = timing.frontPorch + timing.lineSync + (timing.backPorch/ts);
        end
    
%         pause(0.005);
%         fprintf(sprintf('Line %d complete (Image line %d).\n',l,ln));
        l = l+1;
    end
    % now add the vertical retrace segment
    % (should have already accounted for the short version if set to that)
    vbb = ((l-1)*lineSams)+1; %vblank beginning
    vbe = ((l-1)*lineSams)+(vrls*lineSams); % vblank end
    ntsc(1,vbb:vbe) = vBlankEven;
    blankLenV = length(vBlankEven);
    % for even/odd if l is 1, go back to 2nd line, if l is 2, go back to 1
    %  (~(1-1))=1 --> +1 = 2;  (~(2-1))=0 --> +1 = 1
    defy(1,vbb:vbe) = linspace(max(0.5,ln-0.5),(~(iil-1)+1)-0.5,blankLenV);
    defx(1,vbb:vbe) = linspace(xsize,0,blankLenV);

    l = l + vrls; % increment by number of lines just used for vertical retrace
end



imout=imYIQ;


end



% ------------------------------------------------------------------------------------

%%
function [crtaxindex, crtim, rgbcrt, alphacrt, rays] = makeCRT(axid, xsize,ysize,offset,im,interlace)
%CRT Creates an animation of interlaced or progressive CRT raster of an
% image, including animations of the electrons moving through the tube.
% Created by Sean Dabbs for Image Engineering 525.746, Summer 2022
% Note that coordinates are formatted as:
%   -------- > +X
%  |
%  |
%  |
%  V
%  +Y
% so that imagesc() can be easily added

axes(axid);

% offset = 50;

% ------------------------------------------------------------------
%
%  Phosphors
% 
% phosfig = figure(1);

% for debug hard-coding
% xsize = 64;
% ysize = 48;

% clf
hold on


% r = 1;
% apo = r*sqrt(3)/2;


lineHeight = 1; % this will cause phosphors to overflow
lineHeight = (3*ysize)/((3*ysize) + 4); % this shouldn't... but only works for even number of lines?
r = lineHeight*2/3;
apo = r*sqrt(3)/2;
colWidth = apo*6;


hexsc = 0.8; % scaled down so the phosphors won't bunch up against each other
rsc=r*hexsc;
aposc=apo*hexsc; 


vs = lineHeight; % vert step between lines
hs = colWidth; % horizontal step

ho = 2*apo; % horiz offset
vo = 1.5*r; % vert offset

% phosLines = round((ysize/lineHeight)/2)*2;
% phosCols = round((xsize/colWidth)/2)*2;
phosLines = ysize; % phosphor sizes calculated before to line up within CRT
% phosCols = round((xsize/(colWidth+)/2)*2;;
predictedNumCols = (xsize - (7*apo))/(colWidth) + 1; % account for some fudge
phosCols = floor(predictedNumCols*2);
% decide whether to trim one column
if mod(phosCols,2) % if odd
    trimCol = 1;
    phosCols = ceil(phosCols/2);
    predictedPhosphorWidth=(phosCols*colWidth)-(1.5*apo);
    xmargin = xsize-predictedPhosphorWidth;
    ho = ho + (xmargin/2);
else
    trimCol = 0;
    phosCols = round(phosCols/2);
    predictedPhosphorWidth=(phosCols*colWidth)+(1.5*apo);
    xmargin = xsize-predictedPhosphorWidth;
    ho = ho + (xmargin/2);
end



phosVerts = [aposc rsc/2; 0 rsc;-aposc rsc/2; -aposc -rsc/2; 0 -rsc; aposc -rsc/2];
r30 = [sqrt(3)/2 -0.5; 0.5 sqrt(3)/2];
phosVerts30 = phosVerts * r30; % Adds 6 additional vertices, making 12-gon

phosVerts = reshape([phosVerts30.';phosVerts.'],2,[]).';
cox = [-colWidth/6 colWidth/6 0];  % color offset x
coy = [-apo/2 -apo/2 r]; % color offset y

rgbs=["r" "g" "b"]; % RGB strings

% for pv = 1:phosLines % phosphor vertical index
%     % calculate the vertical distance for this triad: will be the same
%     % across this whole line, so saves calcs. Will still update each RGB
%     % phosphor with its own offset from triad center.
%     % ystringperc is fixed length as string, so percentage is floored
%     % note that the triad center's percentage coordinates are assigned, not
%     % individual phosphor, but 'r' 'g' or 'b' is appended.
%     ytriad = (vo+((pv-1)*vs)); % one coordinate y of the center point
%     yup = ytriad + phosVerts(:,2); % y coords for unshifted phosphor; later make 3 and shift each for RGB
%     ytriadperc = ytriad/ysize;
%     ystringperc = sprintf('%04d',floor(min(0.9999999,ytriadperc)*10000)); 
% 
%     for ph = 1:phosCols % phosphor horiz index
%         if ~mod(pv,2) && ph==phosCols % even row, last col
%             0; % skip this column to better fit grid
%         else % proceed as normal
%             xtriad = (mod(1+pv,2)*(colWidth/2))+(ho+((ph-1)*hs)); % one coordinate x of the center point
%             xup = xtriad + phosVerts(:,1); % y coords for unshifted phosphor 
%             xtriadperc = xtriad/xsize;
%             xstringperc = sprintf('%04d',floor(min(0.9999999,xtriadperc)*10000)); 
%             for c = 1:3 % R, G, and B; always 1,2,3
%                 % Create vertex points at appropriate positions
%                 psc = [xup+cox(c) yup+coy(c)]; % phosphor vertices; shifted from center
%                 pgon = polyshape(psc);
%                 % Create name that can later be used to access this phosphor
%                 % from 0 to 9999, representing percent out of 10000 across line or
%                 % screen's height
%                 tid = rgbs(c)+'_'+xstringperc+'_'+ystringperc;
%                 % Do the actual plotting:
%                 phos.(tid) = plot(pgon,'faceColor',rgbs(c),'edgeColor','none','faceAlpha',0.1);
%             end
%         end
%     end
% end

% Matrix version:
% crtscreen = getframe;
% frame2im(crtscreen);
alpmin = 0.1;
alpmax = 1;

% create the RGB phosphor matrix
screenRenderWidth = xsize;%640*4;
screenRenderHeight = ysize;%480*4;
% create a zeros matrix, padded with the specified offset:
rgbcrt = zeros(screenRenderHeight+(2*offset),screenRenderWidth+(2*offset),3);
% assign the RGB matrix to middle of the matrix:
rgbcrt(offset+1:end-offset,offset+1:end-offset,:) = makeCRTRGB(11,3,screenRenderWidth,screenRenderHeight);

% create the alpha matrix:
alphacrt = sum(rgbcrt,3);
alphacrt = alphacrt/max(alphacrt(:)); % scale to 1
alphacrt = alphacrt*alpmin; % scale to the base, "powered off" value

crtim = image(rgbcrt);
crtim.AlphaData = alphacrt;


% ------------------------------------------------------------------
%
%  CRT Frame and Electron Guns
%

% analog timing (not yet used)
% th = ; % horizontal retrace
% tv = ; % vertical retrace
% 
% ts = 1; % sample rate

xs = xsize;
xsh = xs/2; % half
ys = ysize;
ysh = ys/2;

% First, create the "model": 4 sides of the cathode ray tube
crtscale = 0.25; % size of CRT's "back" relative to screen
tubeminx = (xs - (xs*crtscale))/2;
tubemaxx = xs - tubeminx;
tubewidx = xs*crtscale;
tubeminy = (ys - (ys*crtscale))/2;
tubemaxy = ys - tubeminy;
tubewidy = ys*crtscale;

crtdepth = xs; % can be changed depending on "shape" of CRT

rot090z = [0 -1 0
           1  0 0
           0  0 1];
rot090x = [01 00 00
           00 00 -1
           00 01 00];
rot180z = [-1 0 0
           0 -1 0
           0  0 1];
sidey = [...
    tubewidx/2 tubewidy/2 -crtdepth
    xs/2 ys/2 0
    xs/2 -ys/2 0
    tubewidx/2 -tubewidy/2 -crtdepth];
sidex = [...
    tubewidx/2 tubewidy/2 -crtdepth
    xs/2 ys/2 0
    -xs/2 ys/2 0
    -tubewidx/2 tubewidy/2 -crtdepth];

% p2 = side*rot90z

sides = {[sidey] [sidex] [sidey*rot180z] [sidex*rot180z]};
% crtfig = figure(1);
axis off
h = gca;
set(h, 'YDir', 'reverse')
hold on

% piece details
sideAlpha = 0.1;
sideColor = [0.3 0.3 0.3];

for pi=1:1:4
    p = sides{pi};
    fill3(offset+xsh+p(:,1),offset+ysh+p(:,2),p(:,3),sideColor,...
        'FaceAlpha',sideAlpha,'EdgeColor','none');
    sideAlpha = sideAlpha*0.7;
end

% plot the 3 electron guns
gunr = tubewidx*0.05;
gunh = crtdepth*0.1;
[rgunx, rguny, rgunz] = cylinder(gunr);
rgunz=gunh*rgunz;

zdist = -(crtdepth+gunh);

gapo = 1/6;
gr = gapo*2/sqrt(3);
ghexsc = xsize/8; 
grsc=gr*ghexsc; gaposc=gapo*ghexsc;
rgbgunxyz = [(xsize/2)+gaposc (ysize/2)+(grsc/2) zdist
             (xsize/2)-gaposc (ysize/2)+(grsc/2) +zdist
             (xsize/2)+0 (ysize/2)-grsc zdist];
% red green and blue guns
surf(offset+rgunx+rgbgunxyz(1,1), offset+rguny+rgbgunxyz(1,2), rgunz+zdist,...
    'FaceColor',[0.4 0.3 0.3],'EdgeColor',[0.39 0.39 0.39]);
surf(offset+rgunx+rgbgunxyz(2,1), offset+rguny+rgbgunxyz(2,2), rgunz+zdist,...
    'FaceColor',[0.3 0.4 0.3],'EdgeColor',[0.39 0.39 0.39]);
surf(offset+rgunx+rgbgunxyz(3,1), offset+rguny+rgbgunxyz(3,2), rgunz+zdist,...
    'FaceColor',[0.3 0.3 0.4],'EdgeColor',[0.39 0.39 0.39]);

% Add RGB Rays to CRT
rays.('rray') = plot3(offset+[rgbgunxyz(1,1) rgbgunxyz(1,1) 0],...
    offset+[rgbgunxyz(1,2) rgbgunxyz(1,2) 0], ...
    [rgbgunxyz(1,3) rgbgunxyz(1,3)+gunh 0], ...
    'Color','r');
rays.('gray') = plot3(offset+[rgbgunxyz(2,1) rgbgunxyz(2,1) 0],...
    offset+[rgbgunxyz(2,2) rgbgunxyz(2,2) 0], ...
    [rgbgunxyz(2,3) rgbgunxyz(2,3)+gunh 0], ...
    'Color','g');
rays.('bray') = plot3(offset+[rgbgunxyz(3,1) rgbgunxyz(3,1) 0],...
    offset+[rgbgunxyz(3,2) rgbgunxyz(3,2) 0], ...
    [rgbgunxyz(3,3) rgbgunxyz(3,3)+gunh 0], ...
    'Color','b');
% plus a central ray with dotted line to help viewer track movement when
% intensity is 0
rays.('cray') = plot3(offset+[(xsize/2) (xsize/2) 0],...
    offset+[(ysize/2) (ysize/2) 0], ...
    [zdist zdist+gunh 0], ...
    'LineStyle',':','Color',[0.8 0.8 0.8 0.4]);

% line(rgunx(1,:)+(xsize/2), rguny(1,:)+(ysize/2), rgunz(1,:)+zdist);
% line(rgunx(2,:)+(xsize/2), rguny(2,:)+(ysize/2), rgunz(2,:)+zdist);

% plot size and settings
ax = gca;
plotwidth = max([xs ys]);
pwf = plotwidth*0.01; % plot width fraction
plotlimits =[0-pwf plotwidth+pwf];
% xlim(plotlimits);
% ylim(plotlimits);
% zlim(plotlimits);
xlabel('x axis');
ylabel('y axis');
ax.ClippingStyle = '3dbox'; % might help prevent clipping for 3d graphs; alternatives '3dbox' or 'rectangle'
% ax.AmbientLightColor = [1 1 1];
ax.Projection = 'perspective';
ax.DataAspectRatio = [1 1 1];
% ax.InnerPosition = [0 0 1 1]; % should remove margin
axis vis3d;

% get good view
camorbit(0,11*2.5);
camorbit(13*2.5,0,'direction',[0 1 0]);

crtaxindex = 1;

%%

end

%%
function [] = makeIQ(axid,I,Q);
% makes an IQ plot in a vectorscope/phasor style
axes(axid);
hold on;
axisLineColor = [0.6 0.6 0.6];
set(axid, 'color','none');
title('IQ Data (Chrominance)','Color',axisLineColor);
xlabel('In-Phase'); ylabel('Quadrature'); % xlabel('In-Phase','rot',-90);
% axid.FontName = 'FixedWidth';
axid.XAxisLocation = 'origin';
axid.YAxisLocation = 'origin';
axid.XColor = axisLineColor; % should work for axis line, ticks, labels
axid.YColor = axisLineColor; 
% ax.YDir = 'reverse';
axid.LineWidth = 1.0;

% Make outer ring
ringn = 100;
ringt = linspace(0,2*pi,ringn);
ringx = sin(ringt); ringy = cos(ringt);
line(ringx, ringy, 'Color',axisLineColor);
axis equal;
xlim([-1.2 1.2]); ylim([-1.2 1.2]);
% xticks([-1.2 -0.8 -0.6 -0.4 -0.2 0.2 0.4 0.6 0.8 1.2]);
yticks([-0.75 -0.5 -0.25 0.25 0.5 0.75]);
xticks([-0.75 -0.5 -0.25 0.25 0.5 0.75]);
axid.XTickLabel = {};
axid.YTickLabel = {};
% text(0.5, 0.5,'IQ Data (Chrominance)');

% plot([rand rand rand], [rand rand rand]); % temporary

end
%%

function [] = makeStripChart(axid)
axes(axid);
hold on;
axisLineColor = [0.6 0.6 0.6];
xlabel('Time (us)'); ylabel('Voltage (V)');
% axid.FontName = 'FixedWidth';
title('NTSC Signal','Color',axisLineColor);
axid.TitleHorizontalAlignment = 'left';
set(axid, 'color','none');
axid.XColor = axisLineColor; % should work for axis line, ticks, labels
axid.YColor = axisLineColor; 
% plot([1 10 100],[rand rand rand]); % temporary

end

%%


function rgb = makeCRTRGB(circleWidth, padding, width, height)
% creates an RGB matrix of pseudo-hexagonaly-repeated RGB triads,
% similar in appearance to color CRT screens.
% inputs:
% circleWidth: width of each color component's circle, in pixels.
% padding: padding between circles
%
%TODO: shadow mask/Trinitron style option.

% circleWidth = 9; % good defaults
% padding = 1;
radius = floor(circleWidth/2)+1;
pitch = circleWidth+padding;

cirker = makeCircle(circleWidth,radius,padding); % circle kernel

% create matrix that will be repeated as a pattern:
rgbpattern = zeros(pitch*2,(pitch*3),3); 

% create first row:
rgbpattern(1:pitch,1:pitch,1) = cirker;
rgbpattern(1:pitch,pitch+1:2*pitch,2) = cirker;
rgbpattern(1:pitch,(2*pitch)+1:3*pitch,3) = cirker;

% create second row (shalf-shifted version of first row):
rgbpattern(1+pitch:end,1:end,:) = circshift(rgbpattern(1:end/2,:,:),[0 pitch*3/2]);
% figure(); imagesc(255*rgbpattern); axis equal; grid on;

% replicate this pattern to ceate the rgb pattern:
widthReps = ceil(width/(3*pitch));
heightReps = ceil(height/(2*pitch));
rgb = 255*repmat(rgbpattern,heightReps,widthReps);
% figure(); imagesc(rgb); axis equal; grid on;

% smooth the circles with a gaussian filter:
gauker = gaussianKernel(3, 0.5, 0);
rgb = cat(3,conv2(rgb(:,:,1),gauker),...
    conv2(rgb(:,:,2),gauker),...
    conv2(rgb(:,:,3),gauker))/255;
% figure(); imagesc(rgb(:,:,1)); axis equal; grid on;

rgb = rgb(1:height,1:width,:); % crop to specified pixel size


end


function g = gaussianKernel(n, sigma, pad)
[x y] = meshgrid(ceil(-n/2):floor(n/2), ceil(-n/2):floor(n/2));
sizex = size(x);
g = (zeros(sizex+pad));
g(1:sizex,1:size(x)) = exp(-x.^2/(2*sigma^2) - (y.^2/(2*sigma^2)));
% g = g./sum(g(:));
end

function c = makeCircle(n,r,pad) 
[x y] = meshgrid(ceil(-n/2):floor(n/2)+pad, ceil(-n/2):floor(n/2)+pad);
c = uint8(zeros(size(x)));
c=c+uint8((x.^2 + y.^2)<(r^2));
end





