# -*- coding: utf-8 -*-

import os
import time
import numpy
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from data import pad_sequences, batch_yield
from data import read_dictionary, tag2label, label2tag
from utils import get_entity


def predict_one_batch(seqs, stub, dropout=1.0):
    request = predict_pb2.PredictRequest()
    # request.model_spec.name = 'mnist'  # for test
    request.model_spec.name = 'bi_lstm_crf'
    request.model_spec.signature_name = \
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
    request.inputs['word_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(word_ids, shape=[1, seq_len_list[0]]))
    request.inputs['sequence_lengths'].CopyFrom(
        tf.contrib.util.make_tensor_proto(seq_len_list, shape=[1]))
    request.inputs['dropout'].CopyFrom(
        tf.contrib.util.make_tensor_proto(dropout, shape=[]))

    # result_future = stub.Predict.future(request, 2.0)
    # e = result_future.exception()
    # if e:
    #     print e
    #     return
    # result = result_future.result()
    result = stub.Predict(request, 2.0)

    logits_proto = result.outputs['logits']
    # logits_shape = [logits_proto.tensor_shape.dim[i].size
    #                 for i in range(len(logits_proto.tensor_shape.dim))]
    # logits = numpy.array(logits_proto.float_val).reshape(logits_shape)
    logits = tf.contrib.util.make_ndarray(logits_proto)

    transition_params_proto = result.outputs['transition_params']
    # transition_params_shape = [transition_params_proto.tensor_shape.dim[i].size
    #                            for i in range(len(transition_params_proto.tensor_shape.dim))]
    # transition_params = numpy.array(transition_params_proto.float_val).reshape(transition_params_shape)
    transition_params = tf.contrib.util.make_ndarray(transition_params_proto)

    label_list = []
    for logit, seq_len in zip(logits, seq_len_list):
        viterbi, viterbi_score = viterbi_decode(logit[:seq_len], transition_params)
        label_list.append(viterbi)
    return label_list, seq_len_list


word2id = read_dictionary(os.path.join('.', 'data_path', 'word2id.pkl'))


def main(test_sent):
    start_time = time.time()
    channel = implementations.insecure_channel('192.168.1.210', 5075)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    test_sent = list(test_sent.strip())
    test_data = [(test_sent, ['O'] * len(test_sent))]
    label_list = []
    for seqs, labels in batch_yield(test_data, batch_size=64, vocab=word2id, tag2label=tag2label, shuffle=False):
        label_list_, _ = predict_one_batch(seqs, stub)
        label_list.extend(label_list_)
    # label2tag = {}
    # for tag, label in tag2label.items():
    #     label2tag[label] = tag if label != 0 else label
    tag = [label2tag[label] for label in label_list[0]]
    print 'tag', tag
    PER, LOC, ORG = get_entity(tag, test_sent)
    time_used = time.time() - start_time
    print 'tim_used', time_used
    return PER, LOC, ORG


if __name__ == '__main__':
    test_sent = u'隆宇大厦，胡亚东，13517246300'
    PER, LOC, ORG = main(test_sent)
    print 'test_sent', test_sent
    print 'PER', ' '.join(PER)
    print 'LOC', ' '.join(LOC)
    print 'ORG', ' '.join(ORG)
