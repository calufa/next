    cd next
    ./pix2pix-infer/run.sh

    job_name=...
    input_dir=/files/_pix2pix-trainer/${job_name}
    output_dir=/files/_pix2pix-infer/${job_name}
    mkdir -p output_dir
    python pix2pix-tensorflow/pix2pix.py \
      --mode test \
      --checkpoint ${job_name} \
      --input_dir  ${input_dir} \
      --output_dir ${output_dir}

# example

    job_name=test1
    checkpoint=/files/_pix2pix-trainer/${job_name}
    input_dir=/files/_combine-imgs/${job_name}
    output_dir=/files/_pix2pix-infer/${job_name}
    mkdir -p output_dir
    python run.py \
      --mode test \
      --checkpoint ${checkpoint} \
      --input_dir  ${input_dir} \
      --output_dir ${output_dir} \
      --scale_size 256