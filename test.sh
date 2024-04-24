# Function to update a parameter in the YAML file
update_param() {
    param="$1"
    new_value="$2"
    sed -i "s/^\($param\s*:\s*\).*/\1$new_value/" config.yml
}


update_param "dataset\s*:\s*{\s*root_path\s*:\s*.*\s*," "dataset : { root_path : ./datasets/dataset_jaad/bad_weather/"

