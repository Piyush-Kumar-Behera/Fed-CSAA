import tensorflow as tf

def convert_label_swap_dataset(data_orig, a = 5, b = 7):
  test_dict = {
      "pixels": [],
      "label": []
  }

  for var_data in data_orig:
    x = var_data['pixels']
    y = var_data['label']

    if y.numpy() == a:
      new_y = tf.constant(b, shape=(), dtype='int32')
    elif y.numpy() == b:
      new_y = tf.constant(a, shape=(), dtype='int32')
    else:
      new_y = y
    
    test_dict["pixels"].append(x)
    test_dict["label"].append(new_y)
  
  return tf.data.Dataset.from_tensor_slices(test_dict)
