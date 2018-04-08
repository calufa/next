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

# sync file

    job_name=test2
    gcloud compute scp \
    gpu:/home/next/files/_pix2pix-infer/${job_name} ./files/_pix2pix-infer \
      --compress \
      --recurse \
      --project magggenta-176803 \
      --zone us-central1-c

# example

## test2

    model_name=test1
    job_name=test2
    output_name=test3
    checkpoint=/files/_pix2pix-trainer/${model_name}
    input_dir=/files/_combine-imgs/${job_name}
    output_dir=/files/_pix2pix-infer/${output_name}
    mkdir -p output_dir
    python run.py \
      --mode test \
      --checkpoint ${checkpoint} \
      --input_dir  ${input_dir} \
      --output_dir ${output_dir}
