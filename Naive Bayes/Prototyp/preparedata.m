function [ data_with_class ] = preparedata( attr,filename)
ane = attr{1,2}
dane1 = length(attr)
dane22= attr{1:end,2}
%creating data for training 

%length: length(attr) -2 
preinterwal =[] ;
postinterwal = [];
average_interwal = [];
ratio_pre_post = [];
ratio_pre_average = [];

class = []
data_with_class = cell(length(attr) - 2,6)
for i = 2: length(attr) -1
    
    preinterwal = attr{i,1} - attr{i-1,1};
    postinterwal = attr{i+1,1} -attr{i,1};
    average_interwal = (preinterwal + postinterwal)/2;
    ratio_pre_post = preinterwal/postinterwal;
    ratio_pre_average = preinterwal/average_interwal;
    
    data_with_class{i-1,1} = preinterwal;
    data_with_class{i-1,2} = postinterwal;
    data_with_class{i-1,3} = average_interwal;
    data_with_class{i-1,4} = ratio_pre_post;
    data_with_class{i-1,5} = ratio_pre_average;
    if attr{i,2}=='N'     
        data_with_class{i-1,6} = 1;
    elseif attr{i,2} == 'V'
        data_with_class{i-1,6} = 2;
  
    else 
        
        data_with_class{i-1,6} = 3;
    end
    
    
    %data_RR = [data_RR;preinterwal,postinterwal,average_interwal,ratio_pre_post,ratio_pre_average];
    
  
end

filename = 'data_test2.dat';
fid = fopen(filename,'w');
[nrows,ncols] = size(data_with_class);
formatSpec = '%d %d %f %f %f %d\n';
for row = 1:nrows
    fprintf(fid,formatSpec,data_with_class{row,:});
end

% fprintf(fid,'%d %d %f %f %f %s\n',data_with_class{:});
fclose(fid);
% T = cell2table(data_with_class,'VariableNames',{'Preinterwal','Postinterwal','Average','Ratio1','Ratio2','Class'});
% T =table(data_with_class);
% % writetable(T,'tabledata.dat')
end

