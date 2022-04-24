function[] = j2wrot(inpath, outpath)
inpath = char(inpath);
outpath = char(outpath);
time = load(inpath);
[L,~] = size(time(:,1));
for i = 1:L
    dcm = dcmeci2ecef('IAU-2000/2006', time(i, :));
    line_dcm = reshape(dcm', 1, []); 
    if i==1   
        fid = fopen(outpath, 'w');
    else
        fid = fopen(outpath, 'a'); 
    end
    for data = line_dcm(:)
        fprintf(fid, "%f ", data);
    end
    fprintf(fid, "\n");
    fclose(fid);
end
end
