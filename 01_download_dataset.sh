
FILE=neurite-oasis.v1.0.tar
if [ -f "$FILE" ]; then
    echo "$FILE exists."
    echo "oasis3d folder contains data"
else 
    echo "$FILE does not exist."
    wget https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
    mkdir oasis3d
    tar -xvf neurite-oasis.v1.0.tar -C oasis3d
fi


