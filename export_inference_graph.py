"""
Example Usage :
python export_inference_graph.py
--input_type
image_tensor
\
--pipeline_config_path=train_dir/pipeline.config
\
--trained_checkpoint_prefix
train_dir/model.ckpt-1500
\
--output_directory
output_directory

module provides translation of the specified checkpoins into a model with the .pb extension

"""
import sys

import tensorflow.compat.v1 as tf

sys.path.append("D:\\Users\\inet\\Documents\\GitHub\\Sber_test\\models\\research\\slim")

from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                                                  'one of [`image_tensor`, `encoded_image_string_tensor`, '
                                                  '`tf_example`]')
flags.DEFINE_string('input_shape', None,
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('write_inference_graph', False,
                     'If true, writes inference graph to disk.')
flags.DEFINE_string('additional_output_tensor_names', None,
                    'Additional Tensors to output, to be specified as a comma '
                    'separated list of tensor names.')
flags.DEFINE_boolean('use_side_inputs', False,
                     'If True, uses side inputs as well as image inputs.')
flags.DEFINE_string('side_input_shapes', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the shape of the side input tensors to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. A `/` denotes a break, starting the shape of '
                    'the next side input tensor. This flag is required if '
                    'using side inputs.')
flags.DEFINE_string('side_input_types', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the type of the side input tensors. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of types, each of `string`, `integer`, or `float`. '
                    'This flag is required if using side inputs.')
flags.DEFINE_string('side_input_names', None,
                    'If use_side_inputs is True, this explicitly sets '
                    'the names of the side input tensors required by the model '
                    'assuming the names will be a comma-separated list of '
                    'strings. This flag is required if using side inputs.')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = None
    if FLAGS.use_side_inputs:
        side_input_shapes, side_input_names, side_input_types = (
            exporter.parse_side_inputs(
                FLAGS.side_input_shapes,
                FLAGS.side_input_names,
                FLAGS.side_input_types))
    else:
        side_input_shapes = None
        side_input_names = None
        side_input_types = None
    if FLAGS.additional_output_tensor_names:
        additional_output_tensor_names = list(
            FLAGS.additional_output_tensor_names.split(','))
    else:
        additional_output_tensor_names = None
    exporter.export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory, input_shape=input_shape,
        write_inference_graph=FLAGS.write_inference_graph,
        additional_output_tensor_names=additional_output_tensor_names)


if __name__ == '__main__':
    tf.app.run()
