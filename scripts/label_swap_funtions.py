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

def convert_label_swap_dataset_list(data_orig, list_exchange = [(5,7)]):
  test_dict = {
      "pixels": [],
      "label": []
  }

  for var_data in data_orig:
    x = var_data['pixels']
    y = var_data['label']

    done = False
    new_y = None
    for pair_num in list_exchange:
      a, b = pair_num
      if y.numpy() == a:
        new_y = tf.constant(b, shape=(), dtype='int32')
        done = True
        break
      elif y.numpy() == b:
        new_y = tf.constant(a, shape=(), dtype='int32')
        done = True
        break
    if not done:
      new_y = y
    
    test_dict["pixels"].append(x)
    test_dict["label"].append(new_y)
  
  return tf.data.Dataset.from_tensor_slices(test_dict)