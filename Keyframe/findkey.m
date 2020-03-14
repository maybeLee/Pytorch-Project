function findkey
    fileName = 'E:\模式识别课设\行为数据\尝试\Swim';
    fileName2='E:\模式识别课设\行为数据\尝试\new\6';
    list=dir(fullfile(fileName));
    numberoffile=size(list,1);
    j=0;
    for i=3:numberoffile
        obj = VideoReader(fullfile(fileName,list(i).name));
        numFrames = obj.NumberOfFrames;
        cha=zeros(1,numFrames-1);
        for k=2:numFrames
            lastframe=read(obj,k-1);
            frame=read(obj,k);
            grayframe=rgb2gray(frame);
            lastgrayframe=rgb2gray(lastframe);
            difgrayFrame= grayframe - lastgrayframe;
            cha(k-1)=mean(difgrayFrame(:));
        end
        for k=2:numFrames-2
            if cha(k)>cha(k-1)&&cha(k)>cha(k+1)
                j=j+1;
                frame=read(obj,k+1);
                imwrite(frame,fullfile(fullfile(fileName2,strcat(num2str(i-2))),strcat(num2str(j),'.png')));
            end
        end
    end
        
    