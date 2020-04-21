def split_data(input, output, validation_percentage=0.1):
    num_sets = output.shape[0]
    num_validation = int(num_sets * validation_percentage)
    return (input[:-num_validation], output[:-num_validation]), (input[-num_validation:], output[-num_validation:])
