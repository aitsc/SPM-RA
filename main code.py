import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import sys
import heapq
import pickle
import time
import numpy as np

from bert_serving.client import BertClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from pprint import pprint

class 标题摘要_tf模型:
    def __init__(self,模型参数,初始词向量l=None,可视化地址=None,取消可视化=False,BERT句词向量目录=None):
        assert 模型参数,'缺少模型参数, 是dict或要读取的模型地址.'
        self._可视化w = None
        self._保存模型地址_saver表 = {}
        if isinstance(模型参数,dict):
            assert 'haveTrainingSteps' not in 模型参数,'参数不能包含"haveTrainingSteps"!'
            g, init, self._词_序号d = self._构建计算图(模型参数,初始词向量l)
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数['显存占用比']
            self._sess=tf.Session(graph=g,config=tf_config)
            self._sess.run(init)
            if 可视化地址 and not 取消可视化:
                self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
            self._模型参数d = 模型参数.copy()
            self._模型参数d['haveTrainingSteps']=0
        else:
            self._sess, self._模型参数d, self._词_序号d = self._读取模型(模型参数)
            if not 取消可视化:
                if not 可视化地址: 可视化地址 = 模型参数
                try:
                    self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
                except:
                    print('模型不含可视化!')
        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            print('开启一个BertClient(需要先开启BertServer)...')
            self._BertClient = BertClient()
            self._BERT句词向量目录 = BERT句词向量目录
            self._BERT句词向量d = {}
            if self._模型参数d['BERT句向量存取加速']:
                try:
                    print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + ':读取BERT句向量...', end='')
                    sys.stdout.flush()
                    startTime = time.time()
                    with open(BERT句词向量目录.encode('utf-8'),'rb') as r:
                        while True:
                            try: self._BERT句词向量d.update([pickle.load(r)])
                            except: break
                    print('%.2fm' % ((time.time() - startTime) / 60))
                except:
                    with open(BERT句词向量目录.encode('utf-8'), 'wb'): ...
        else:
            self._BertClient = None

    def _读取模型(self,模型地址):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        startTime=time.time()
        with open((模型地址+'.parms').encode('utf-8'),'r',encoding='utf-8') as r: 模型参数d = eval(r.read())
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数d['显存占用比']
        sess = tf.Session(config=tf_config)
        saver = tf.train.import_meta_graph(模型地址+'.meta')
        saver.restore(sess,模型地址)
        with open((模型地址+'.w_num_map').encode('utf-8'),'rb') as r: 词_序号d = pickle.load(r)
        print('%.2fm' % ((time.time() - startTime) / 60))
        pprint(模型参数d)
        return sess, 模型参数d, 词_序号d

    def _构建计算图(self,模型参数d,初始词向量l):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        startTime=time.time()
        词_序号d={'':0}
        序号_词d={0:''}
        词向量矩阵l=[[0.]*模型参数d['embedding_dim']]
        graph = tf.Graph()
        with graph.as_default():
            if '使用BERT' not in 模型参数d or not 模型参数d['使用BERT']:
                词数上限 = 模型参数d['词数上限']
                assert 词数上限,'需要给定词数上限!'
                词数上限 += 1
                if not 初始词向量l: 初始词向量l = []
                if len(初始词向量l) > 词数上限: 词数上限 = len(初始词向量l) + 1
                for i,(词,向量) in enumerate(初始词向量l):
                    i+=1
                    词_序号d[词]=i
                    序号_词d[i]=词
                    词向量矩阵l.append(向量)
                for i in range(len(词向量矩阵l),词数上限):
                    if 模型参数d['词向量固定值初始化'] and -1<=模型参数d['词向量固定值初始化']<=1:
                        词向量矩阵l.append([模型参数d['词向量固定值初始化']]*模型参数d['embedding_dim'])
                    else:
                        词向量矩阵l.append([random.uniform(-1, 1) for i in range(模型参数d['embedding_dim'])])
                if 模型参数d['固定词向量']: w_embed = tf.constant(词向量矩阵l, name='w_embed')
                else: w_embed = tf.get_variable('w_embed', initializer=词向量矩阵l)
                tf.constant(词数上限,name='word_count_limit')
                with tf.variable_scope('w_embed_summary'):
                    w_embed_均值 = tf.reduce_mean(w_embed)
                    w_embed_方差 = tf.reduce_mean(tf.square(w_embed - w_embed_均值))
                    w_embed_标准差 = tf.sqrt(w_embed_方差)
                    tf.summary.histogram('w_embed_weights', w_embed)
                    tf.summary.scalar('w_embed_weights_E', w_embed_均值)
                    tf.summary.scalar('w_embed_weights_S', w_embed_标准差)
                title_p = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']],name='title_p')
                abstract_p = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']],name='abstract_p')
                title_n = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']],name='title_n')
                abstract_n = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']],name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)
                title_p_n = tf.nn.embedding_lookup(w_embed, title_p_n)
                abstract_p_n = tf.nn.embedding_lookup(w_embed, abstract_p_n)
            else:
                title_p = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='title_p')
                abstract_p = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='abstract_p')
                title_n = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='title_n')
                abstract_n = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)
            title_len_p = tf.placeholder(tf.int32, [None],name='title_len_p')
            abstract_len_p = tf.placeholder(tf.int32, [None],name='abstract_len_p')
            title_len_n = tf.placeholder(tf.int32, [None],name='title_len_n')
            abstract_len_n = tf.placeholder(tf.int32, [None],name='abstract_len_n')
            title_len = tf.concat([title_len_p, title_len_n],0)
            abstract_len = tf.concat([abstract_len_p, abstract_len_n],0)
            if '使用BERT' in 模型参数d and 模型参数d['使用BERT']:
                if not 模型参数d['使用[SEP]']:
                    title_len -= 1
                    abstract_len -= 1
                    mask_t = tf.sequence_mask(title_len, 模型参数d['BERT_maxlen'])
                    mask_a = tf.sequence_mask(abstract_len, 模型参数d['BERT_maxlen'])
                    零一 = tf.constant([[0.]*模型参数d['BERT_embedding_dim'], [1.]*模型参数d['BERT_embedding_dim']])
                    mask_t = tf.nn.embedding_lookup(零一, tf.to_int32(mask_t))
                    mask_a = tf.nn.embedding_lookup(零一, tf.to_int32(mask_a))
                    title_p_n *= mask_t
                    abstract_p_n *= mask_a
                if not 模型参数d['使用[CLS]']:
                    title_len -= 1
                    abstract_len -= 1
                    title_p_n = tf.concat([title_p_n[:,1:,:], title_p_n[:,:1,:]*0], axis=1)
                    abstract_p_n = tf.concat([abstract_p_n[:,1:,:], abstract_p_n[:,:1,:]*0], axis=1)
            if 模型参数d['词向量tanh']:
                title_p_n = tf.tanh(title_p_n)
                abstract_p_n = tf.tanh(abstract_p_n)
            if 模型参数d['词向量微调']:
                title_p_n = self._词向量微调层(title_p_n)['outputs']
                输出 = self._词向量微调层(abstract_p_n)
                abstract_p_n = 输出['outputs']
                tf.summary.histogram('embedding_weights', 输出['weights'])
                tf.summary.histogram('embedding_biases', 输出['biases'])
            assert 模型参数d['使用LSTM'] or 模型参数d['使用CNN'], '至少使用一种训练模型!'
            keep_prob_LSTM = tf.placeholder(tf.float32,name='keep_prob_LSTM')
            keep_prob_CNN = tf.placeholder(tf.float32,name='keep_prob_CNN')
            输出t = {'outputs_all': title_p_n}
            输出a = {'outputs_all': abstract_p_n}
            if 模型参数d['共享参数']:
                with tf.variable_scope('share_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出t = self._biLSTM(title_p_n, title_len, 模型参数d,keep_prob_LSTM)
                        输出a = self._biLSTM(abstract_p_n, abstract_len, 模型参数d,keep_prob_LSTM,False)
                    if 模型参数d['使用CNN']:
                        输出t = self._CNN2d(输出t['outputs_all'], 模型参数d,keep_prob_CNN)
                        输出a = self._CNN2d(输出a['outputs_all'], 模型参数d,keep_prob_CNN,False)
            else:
                with tf.variable_scope('title_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出t = self._biLSTM(title_p_n,title_len,模型参数d,keep_prob_LSTM)
                    if 模型参数d['使用CNN']:
                        输出t = self._CNN2d(输出t['outputs_all'], 模型参数d,keep_prob_CNN)
                with tf.variable_scope('abstract_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出a =self._biLSTM(abstract_p_n,abstract_len,模型参数d,keep_prob_LSTM)
                    if 模型参数d['使用CNN']:
                        输出a = self._CNN2d(输出a['outputs_all'], 模型参数d,keep_prob_CNN)
            outputs_t = 输出t['outputs']
            outputs_a = 输出a['outputs']
            with tf.variable_scope('sim_f'):




                outputs_t = tf.nn.l2_normalize(outputs_t, 1, name='title_l2vec')
                outputs_a = tf.nn.l2_normalize(outputs_a, 1, name='abstract_l2vec')
                sim_p_n = tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1,name='sim_p_n')
            with tf.variable_scope('sim_result'):
                sim_p, sim_n = tf.split(sim_p_n, num_or_size_splits=2, axis=0)
                正例相似度均值 = tf.reduce_mean(sim_p)
                负例相似度均值 = tf.reduce_mean(sim_n)
                正例相似度方差 = tf.reduce_mean(tf.square(sim_p - 正例相似度均值))
                负例相似度方差 = tf.reduce_mean(tf.square(sim_n - 负例相似度均值))
                正例相似度标准差 = tf.sqrt(正例相似度方差)
                负例相似度标准差 = tf.sqrt(负例相似度方差)
                tf.summary.scalar('E_positive', 正例相似度均值)
                tf.summary.scalar('E_negative', 负例相似度均值)
                tf.summary.scalar('S_positive', 正例相似度标准差)
                tf.summary.scalar('S_negative', 负例相似度标准差)
            with tf.variable_scope('loss_f'):
                负正差 = sim_n - sim_p + 模型参数d['margin']
                # 负正差 = tf.maximum(0., sim_n - sim_p + 模型参数d['margin'])
                # 一正差 = (1. - sim_p) * 模型参数d['margin']
                loss_op = tf.reduce_sum(tf.maximum(0.0, 负正差),name='loss_op')
                # loss_op = tf.reduce_sum(对数负正差,name='loss_op')
                # loss_op = tf.reduce_sum(tf.exp(负正差 + 一正差)-1, name='loss_op')
                # loss_op = tf.reduce_sum(tf.maximum(0., tf.exp(负正差 + 一正差)-1), name='loss_op')
                tf.summary.scalar('loss', loss_op)
                tf.summary.scalar('loss_max', tf.reduce_max(负正差))
                对错二分_loss = tf.to_float(tf.greater(0., 负正差), name='right_error_list')
            with tf.variable_scope('acc_f'):
                对错二分 = tf.to_float(tf.greater(sim_p, sim_n), name = 'right_error_list')
                accuracy_op=tf.reduce_mean(对错二分,name='accuracy_op')
                tf.summary.scalar('accuracy', accuracy_op)
            with tf.variable_scope('index'):
                tf.summary.histogram('P_sub_N_sim', sim_p - sim_n)
            global_step = tf.Variable(0)
            if 0<模型参数d['AdamOptimizer']<1:
                optimizer = tf.train.AdamOptimizer(learning_rate=模型参数d['AdamOptimizer'])
            else:
                学习率最小值 = 模型参数d['learning_rate']/模型参数d['学习率最小值倍数']
                learning_rate = tf.maximum(学习率最小值, tf.train.exponential_decay(
                    learning_rate = 模型参数d['learning_rate'],
                    global_step = global_step,
                    decay_steps = 模型参数d['学习率衰减步数'],
                    decay_rate = 模型参数d['学习率衰减率'],
                ))
                tf.summary.scalar('learning_rate', learning_rate)
                optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # train_op = optimizer.minimize(loss_op, global_step=global_step, name='train_op')
            grads_and_vars = optimizer.compute_gradients(loss_op)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')
            with tf.variable_scope('v_grad'):
                for g, v in grads_and_vars:
                    if g is not None:
                        tf.summary.histogram("{}".format(v.name), g)
                        # tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
            init = tf.global_variables_initializer()
            tf.summary.merge_all(name='merged_op')

        print('%.2fm'%((time.time()-startTime)/60))
        return graph,init,词_序号d

    def _biLSTM(self,x, sequence_length, 模型参数d, keep_prob=1., 可视化=True):
        各隐藏层数l = 模型参数d['biLSTM_各隐层数']
        outputs = x
        biLSTM池化方法 = eval(模型参数d['biLSTM池化方法'])
        输出 = {}
        for i,隐层数 in enumerate(各隐藏层数l):
            input_keep_prob = keep_prob
            output_keep_prob = keep_prob
            if i>0:
                input_keep_prob = 1
            with tf.variable_scope("biLSTM%dl"%i):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                输出['lstm_fw_cell%dL'%(i+1)] = lstm_fw_cell
                输出['lstm_bw_cell%dL'%(i+1)] = lstm_bw_cell
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_fw_cell,
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                )
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_bw_cell,
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                )
                outputs, output_states_fw, _ = rnn.stack_bidirectional_dynamic_rnn([lstm_fw_cell], [lstm_bw_cell], outputs,dtype=tf.float32,sequence_length=sequence_length)
                前向隐层 = output_states_fw[-1][-1]
                反向隐层 = tf.split(tf.squeeze(outputs[:, :1, :], [1]), 2, 1)[1]
                if biLSTM池化方法 != tf.concat:
                    outputs = biLSTM池化方法(tf.split(outputs, 2, 2), 0)
                if i + 1 >= len(各隐藏层数l):
                    输出['outputs_all'] = outputs
                    f = eval(模型参数d['LSTM序列池化方法'])
                    if f:
                        if f == tf.reduce_mean:
                            outputs = tf.reduce_sum(outputs, 1) / sequence_length
                        elif f == tf.reduce_max:
                            outputs = f(outputs, 1)
                        else:
                            assert False,'不支持其他池化方式!'
                    else:
                        if biLSTM池化方法 != tf.concat:
                            outputs = biLSTM池化方法([前向隐层, 反向隐层], 0)
                        else:
                            outputs = tf.concat([前向隐层, 反向隐层], 1)
                    输出['outputs']=outputs
        if 可视化:
            for name, cell in 输出.items():
                if 'cell' not in name: continue
                tf.summary.histogram(name + '-w', cell.weights[0])
                tf.summary.histogram(name + '-b', cell.weights[1])
        return 输出

    def _CNN2d(self, x, 模型参数d, keep_prob=1., 可视化=True):
        num_filters = 模型参数d['num_filters']
        filter_sizes = 模型参数d['filter_sizes']
        embedding_dim = int(x.shape[2])
        x = tf.expand_dims(x, -1)

        输出 = {}
        with tf.variable_scope("CNN2d"):
            pooled = []
            for filter_size in filter_sizes:
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                w = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='CNN_cell%dfs_b'%filter_size)
                b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_filters]), name='CNN_cell%dfs_w'%filter_size)
                输出['CNN_cell%dfs_w'%filter_size] = w
                输出['CNN_cell%dfs_b'%filter_size] = b
                conv = tf.nn.conv2d(x, w, strides=[1]*4, padding='VALID')
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                max_pool = tf.nn.max_pool(h, ksize=[1, h.shape[1], 1, 1], strides=[1] * 4, padding='VALID')
                if 模型参数d['CNN输出层tanh']:
                    max_pool = tf.tanh(max_pool)
                pooled.append(max_pool)
            num_filters_total = num_filters * len(filter_sizes)
            pooled = tf.reshape(tf.concat(pooled, 3), [-1, num_filters_total])
            pooled = tf.nn.dropout(pooled, keep_prob)
            输出['outputs'] = pooled
        if 可视化:
            for name, cell in 输出.items():
                if 'cell' not in name: continue
                tf.summary.histogram(name, cell)
        return 输出

    def _词向量微调层(self, 句子词张量):
        with tf.variable_scope("fine_tune_embedding",reuse=tf.AUTO_REUSE):
            embedding_dim = int(句子词张量.shape[-1])
            weights = tf.get_variable('weights',initializer=tf.random_normal([embedding_dim, embedding_dim]))
            biases = tf.get_variable('biases',initializer=tf.random_normal([embedding_dim]))
            outputs = tf.einsum('ijk,kl->ijl', 句子词张量, weights) + biases
            outputs = tf.tanh(outputs)
            输出 = {}
            输出['weights'] = weights
            输出['biases'] = biases
            输出['outputs'] = outputs
        return 输出

    def _构建BERT词向量(self,title_p, abstract_p, title_n, abstract_n,
           title_len_p, abstract_len_p, title_len_n, abstract_len_n):
        assert self._BertClient, '没有BertClient!'
        BERT_maxlen = self._模型参数d['BERT_maxlen']
        所有句子l = [' '.join(i[:BERT_maxlen]) for i in title_p + abstract_p + title_n + abstract_n]
        if self._模型参数d['BERT句向量存取加速']:
            句子向量张量l = []
            句子序号_位置d = {}
            待计算句子l = []
            for i,句子 in enumerate(所有句子l):
                if 句子 in self._BERT句词向量d:
                    句子向量张量l.append(self._BERT句词向量d[句子])
                else:
                    句子序号_位置d[len(句子序号_位置d)] = i
                    待计算句子l.append(句子)
                    句子向量张量l.append(None)
            if 待计算句子l:
                计算_句子向量张量l = self._BertClient.encode(待计算句子l)
                w = open(self._BERT句词向量目录.encode('utf-8'), 'ab')
                for i,句子向量张量 in enumerate(计算_句子向量张量l):
                    序号 = 句子序号_位置d[i]
                    句子向量张量l[序号] = 句子向量张量
                    句子 = 所有句子l[序号]
                    self._BERT句词向量d[句子] = 句子向量张量
                    pickle.dump((句子,句子向量张量), w)
                w.close()
            句子向量张量l = np.array(句子向量张量l)
        else:
            句子向量张量l = self._BertClient.encode(所有句子l)
        句子长度向量l = title_len_p + abstract_len_p + title_len_n + abstract_len_n
        for i in range(len(句子长度向量l)):
            句子长度向量l[i] += 2
            if 句子长度向量l[i] > BERT_maxlen:
                句子长度向量l[i] = BERT_maxlen
        句子长度向量l = np.array(句子长度向量l)

        组数 = bool(title_p) + bool(abstract_p) + bool(title_n) + bool(abstract_n)
        return np.split(句子向量张量l, 组数, axis=0) + np.split(句子长度向量l, 组数, axis=0)

    def 预_编号与填充(self,句_词变矩阵l,isTitle,加入新词=True):
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        assert not isinstance(句_词变矩阵l[0][0], list), '数据格式错误!(句_词变矩阵l[0][0]=%s)'%str(句_词变矩阵l[0][0])
        新词数, 新词s, 加入新词s = 0, set(), set()
        最大长度 = self._模型参数d['title_maxlen'] if isTitle else self._模型参数d['abstract_maxlen']
        长度l = []

        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            句_词矩阵l = []
            for 句子 in 句_词变矩阵l:
                句_词矩阵l.append(句子[:最大长度])
                长度l.append(len(句_词矩阵l[-1]))
        else:
            句_词矩阵l = [[0]*最大长度 for i in range(len(句_词变矩阵l))]
            词数上限=self._sess.run(self._sess.graph.get_tensor_by_name('word_count_limit:0'))
            for i,句子 in enumerate(句_词变矩阵l):
                长度 = 0
                for j,词 in enumerate(句子):
                    if j >= 最大长度:
                        break
                    if 词 in self._词_序号d:
                        句_词矩阵l[i][长度] = self._词_序号d[词]
                        长度 += 1
                    else:
                        新词数 += 1
                        新词s.add(词)
                        if 词数上限 > len(self._词_序号d) and 加入新词 and self._模型参数d['可加入新词']:
                            加入新词s.add(词)
                            序号 = len(self._词_序号d) + 1
                            self._词_序号d[词] = 序号
                            句_词矩阵l[i][长度] = 序号
                            长度 += 1
                长度l.append(长度)
        return 句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s

    def 预_编号与填充_批量(self,多_句_词变矩阵l,isTitle向量,加入新词=True):
        all新词数, all新词s, all加入新词s = 0, set(), set()
        多_句_词矩阵l = []
        多_长度l = []
        for 句_词变矩阵l,isTitle in zip(多_句_词变矩阵l,isTitle向量):
            句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s = self.预_编号与填充(句_词变矩阵l, isTitle, 加入新词)
            多_句_词矩阵l.append(句_词矩阵l)
            多_长度l.append(长度l)
            all新词数 += 新词数
            all新词s |= 新词s
            all加入新词s |= 加入新词s
        return 多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s

    def 训练(self,title_p, abstract_p, title_n, abstract_n,
           title_len_p, abstract_len_p, title_len_n, abstract_len_n,
           记录过程=True,记录元数据=False,合并之前训练错误的数据=None):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        assert not isinstance(title_p[0][0],list),'数据格式错误!(title_p[0][0])'
        assert len(title_p)==len(title_len_p),'数据格式错误!(len(title_p)==len(title_len_p))'

        if 合并之前训练错误的数据:
            title_p += 合并之前训练错误的数据[0]
            abstract_p += 合并之前训练错误的数据[1]
            title_n += 合并之前训练错误的数据[2]
            abstract_n += 合并之前训练错误的数据[3]
            title_len_p += 合并之前训练错误的数据[4]
            abstract_len_p += 合并之前训练错误的数据[5]
            title_len_n += 合并之前训练错误的数据[6]
            abstract_len_n += 合并之前训练错误的数据[7]

        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            title_p, abstract_p, title_n, abstract_n, title_len_p, abstract_len_p, title_len_n, abstract_len_n = self._构建BERT词向量(
                title_p, abstract_p, title_n, abstract_n, title_len_p, abstract_len_p, title_len_n, abstract_len_n)
        else:
            title_maxlen = self._模型参数d['title_maxlen']
            abstract_maxlen = self._模型参数d['abstract_maxlen']
            assert len(title_p[0]) == title_maxlen, '标题长度与参数不匹配!(%d==%d)' % (len(title_p[0]), title_maxlen)
            assert len(abstract_n[0]) == abstract_maxlen, '摘要长度与参数不匹配!(%d==%d)' % (len(abstract_n[0]), abstract_maxlen)

        feed_dict = {'title_p:0': title_p, 'abstract_p:0': abstract_p,
                     'title_n:0': title_n, 'abstract_n:0': abstract_n,
                     'title_len_p:0': title_len_p, 'abstract_len_p:0': abstract_len_p,
                     'title_len_n:0': title_len_n, 'abstract_len_n:0': abstract_len_n,
                     'keep_prob_LSTM:0': self._模型参数d['LSTM_dropout'],
                     'keep_prob_CNN:0': self._模型参数d['CNN_dropout']}

        self._模型参数d['haveTrainingSteps'] += 1
        训练op = {}
        训练op['loss']=self._sess.graph.get_tensor_by_name('loss_f/loss_op:0')
        训练op['对错二分']=self._sess.graph.get_tensor_by_name('acc_f/right_error_list:0')
        训练op['acc']=self._sess.graph.get_tensor_by_name('acc_f/accuracy_op:0')
        训练op['train']=self._sess.graph.get_operation_by_name('train_op')
        训练op['loss对错二分']=self._sess.graph.get_tensor_by_name('loss_f/right_error_list:0')
        if self._可视化w and 记录过程:
            训练op['merged'] = self._sess.graph.get_tensor_by_name('merged_op/merged_op:0')
            if 记录元数据:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None
            训练结果d = self._sess.run(训练op, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            if 记录元数据:
                self._可视化w.add_run_metadata(run_metadata, 'step%d' % self._模型参数d['haveTrainingSteps'])
            self._可视化w.add_summary(训练结果d['merged'], self._模型参数d['haveTrainingSteps'])
        else:
            训练结果d = self._sess.run(训练op, feed_dict=feed_dict)

        训练错误的数据 = [[] for i in range(8)]
        for i,true in enumerate(训练结果d['对错二分']):
            if true: continue
            训练错误的数据[0].append(title_p[i])
            训练错误的数据[1].append(abstract_p[i])
            训练错误的数据[2].append(title_n[i])
            训练错误的数据[3].append(abstract_n[i])
            训练错误的数据[4].append(title_len_p[i])
            训练错误的数据[5].append(abstract_len_p[i])
            训练错误的数据[6].append(title_len_n[i])
            训练错误的数据[7].append(abstract_len_n[i])

        损失函数大于0的数据 = [[] for i in range(8)]
        for i,true in enumerate(训练结果d['loss对错二分']):
            if true: continue
            损失函数大于0的数据[0].append(title_p[i])
            损失函数大于0的数据[1].append(abstract_p[i])
            损失函数大于0的数据[2].append(title_n[i])
            损失函数大于0的数据[3].append(abstract_n[i])
            损失函数大于0的数据[4].append(title_len_p[i])
            损失函数大于0的数据[5].append(abstract_len_p[i])
            损失函数大于0的数据[6].append(title_len_n[i])
            损失函数大于0的数据[7].append(abstract_len_n[i])

        输出 = {
            '损失函数值':训练结果d['loss'],
            '精确度':训练结果d['acc'],
            '训练错误的数据':训练错误的数据,
            '损失函数大于0的数据':损失函数大于0的数据,
            '实际训练数据大小':len(title_p),
        }
        return 输出

    def 测试(self,批次列表,topN=1,batch_size=None,可加新词=False):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        assert not isinstance(批次列表[0][2][0][0],list),'数据格式错误!'
        批次列表_实战=[]
        每批次正例数l=[]
        batch_top_FP, batch_top_FN, batch_top_TP, batch_top_TN = [], [], [], []
        batch_top_平均相似度 = []

        for j in 批次列表:
            if isinstance(j[0][0], list):
                每批次正例数l.append(len(j[0]))
                标题 = j[0]+j[2]
                摘要 = j[1]
            else:
                每批次正例数l.append(len(j[1]))
                摘要 = j[1]+j[2]
                标题 = j[0]
            批次列表_实战.append([标题,摘要])
        批次_相似度矩阵, all_新词个数, all_新词s, all_加入新词s = self.实战(批次列表=批次列表_实战, batch_size=batch_size, 可加新词=可加新词)
        for i in range(len(批次_相似度矩阵)):
            正例数=每批次正例数l[i]
            序号_得分逆序l = heapq.nlargest(topN, enumerate(批次_相似度矩阵[i]), key=lambda t: t[1])
            预测结果l=[1 if i[0] < 正例数 else 0 for i in 序号_得分逆序l]
            top_FP, top_FN, top_TP, top_TN = [], [], [], []
            top_平均相似度 = []
            预测正确数=0
            for j in range(topN):
                预测正确数+=预测结果l[j]
                top_FP.append(j+1-预测正确数)
                top_FN.append(正例数-预测正确数)
                top_TP.append(预测正确数)
                top_TN.append(len(批次_相似度矩阵[i])-(j+1)-(正例数-预测正确数))
                top_平均相似度.append(sum([k for _,k in 序号_得分逆序l[:j+1]])/(j+1))
            batch_top_FP.append(top_FP)
            batch_top_FN.append(top_FN)
            batch_top_TP.append(top_TP)
            batch_top_平均相似度.append(top_平均相似度)
        batch_top_FP=np.array(batch_top_FP)
        batch_top_FN=np.array(batch_top_FN)
        batch_top_TP=np.array(batch_top_TP)
        batch_top_平均相似度=np.array(batch_top_平均相似度)

        top_macroP = (batch_top_TP/(batch_top_TP+batch_top_FP)).mean(axis=0)
        top_macroR = (batch_top_TP/(batch_top_TP+batch_top_FN)).mean(axis=0)
        top_macroF1 = 2*top_macroP*top_macroR/(top_macroP+top_macroR)
        top_平均相似度 = batch_top_平均相似度.mean(axis=0)
        top指标={
            'macro_P':list(top_macroP),
            'macro_R':list(top_macroR),
            'macro_F1':list(top_macroF1),
            '平均相似度':list(top_平均相似度),
        }
        return top指标,all_新词个数, all_新词s, all_加入新词s

    def 实战(self,批次列表,batch_size=None,可加新词=False):
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        标题_序号d = {}
        摘要_序号d = {}
        批次列表_序号 = []
        批次_相似度矩阵 = []

        for j in 批次列表:
            if isinstance(j[0][0], list):
                标题l = j[0]
                摘要l = [j[1]]
            else:
                摘要l = j[1]
                标题l = [j[0]]
            标题序号l = []
            摘要序号l = []
            for 标题 in 标题l:
                标题 = tuple(标题)
                if 标题 not in 标题_序号d:
                    标题_序号d[标题] = len(标题_序号d)
                标题序号l.append(标题_序号d[标题])
            for 摘要 in 摘要l:
                摘要 = tuple(摘要)
                if 摘要 not in 摘要_序号d:
                    摘要_序号d[摘要] = len(摘要_序号d)
                摘要序号l.append(摘要_序号d[摘要])
            if isinstance(j[0][0], list):
                批次列表_序号.append([标题序号l, 摘要序号l[0]])
            else:
                批次列表_序号.append([标题序号l[0], 摘要序号l])

        标题l = [i for i,_ in sorted(标题_序号d.items(),key=lambda t:t[1])]
        摘要l = [i for i,_ in sorted(摘要_序号d.items(),key=lambda t:t[1])]
        标题_摘要cos矩阵, all_新词个数, all_新词s, all_加入新词s = self.标题摘要相似度计算(标题l,摘要l,batch_size,可加新词)
        for i in 批次列表_序号:
            批次_相似度矩阵.append([])
            if isinstance(i[0], list):
                摘要序号 = i[1]
                for 标题序号 in i[0]:
                    批次_相似度矩阵[-1].append(标题_摘要cos矩阵[标题序号][摘要序号])
            else:
                标题序号 = i[0]
                for 摘要序号 in i[1]:
                    批次_相似度矩阵[-1].append(标题_摘要cos矩阵[标题序号][摘要序号])
        return 批次_相似度矩阵, all_新词个数, all_新词s, all_加入新词s

    def 标题摘要相似度计算(self,标题l,摘要l,batch_size,可加新词=False,保留进度=True):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        all_新词个数, all_新词s, all_加入新词s = 0, set(), set()
        title_maxlen=self._模型参数d['title_maxlen']
        abstract_maxlen=self._模型参数d['abstract_maxlen']
        函数名=self.__class__.__name__ + '.' + sys._getframe().f_code.co_name

        title_p_n, title_len, 新词数, 新词s, 加入新词s = self.预_编号与填充(标题l, True, 加入新词=可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s
        abstract_p_n, abstract_len, 新词数, 新词s, 加入新词s = self.预_编号与填充(摘要l, False, 加入新词=可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s

        feed_dict = {
            'keep_prob_LSTM:0': 1.,
            'keep_prob_CNN:0': 1.
        }
        标题l2向量op = self._sess.graph.get_tensor_by_name('sim_f/title_l2vec:0')
        标题l2向量l=[]
        for i in tqdm(range(0,len(title_p_n),batch_size), desc=函数名+'(标题)',leave=保留进度):
            batch, batch_len = title_p_n[i:i + batch_size], title_len[i:i + batch_size]
            batch_空 = np.zeros([0, title_maxlen], np.int32)
            if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
                batch, batch_len = self._构建BERT词向量(batch, [], [], [], batch_len, [], [], [])
                batch_空 = np.zeros([0, self._模型参数d['BERT_maxlen'], self._模型参数d['BERT_embedding_dim']], np.float32)
            标题l2向量 = self._sess.run(标题l2向量op,
                                      feed_dict={'title_p:0': batch,
                                                 'title_n:0': batch_空,
                                                 'title_len_p:0': batch_len,
                                                 'title_len_n:0': np.zeros([0], np.int32),
                                                 **feed_dict})
            标题l2向量l+=list(标题l2向量)
        摘要l2向量op = self._sess.graph.get_tensor_by_name('sim_f/abstract_l2vec:0')
        摘要l2向量l=[]

        for i in tqdm(range(0,len(abstract_p_n),batch_size), desc=函数名+'(摘要)',leave=保留进度):
            batch, batch_len = abstract_p_n[i:i + batch_size], abstract_len[i:i + batch_size]
            batch_空 = np.zeros([0, abstract_maxlen], np.int32)
            if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
                batch, batch_len = self._构建BERT词向量(batch, [], [], [], batch_len, [], [], [])
                batch_空 = np.zeros([0, self._模型参数d['BERT_maxlen'], self._模型参数d['BERT_embedding_dim']], np.float32)
            摘要l2向量 = self._sess.run(摘要l2向量op,
                                    feed_dict={'abstract_p:0': batch,
                                               'abstract_n:0': batch_空,
                                               'abstract_len_p:0': batch_len,
                                               'abstract_len_n:0': np.zeros([0], np.int32),
                                               **feed_dict})
            摘要l2向量l+=list(摘要l2向量)
        标题_摘要cos矩阵=np.dot(np.array(标题l2向量l),np.array(摘要l2向量l).T)
        return 标题_摘要cos矩阵, all_新词个数, all_新词s, all_加入新词s

    def 论文相似度计算(self,论文xl,论文yl,batch_size,max0_avg1_min2_sum3_t4_a5=1,可加新词=False):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        all_新词个数, all_新词s, all_加入新词s = 0, set(), set()

        标题xl, 摘要xl = 论文xl
        标题yl, 摘要yl = 论文yl
        标题_摘要cos矩阵xy, 新词数, 新词s, 加入新词s = self.标题摘要相似度计算(标题xl, 摘要yl, batch_size,可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s
        标题_摘要cos矩阵yx, 新词数, 新词s, 加入新词s = self.标题摘要相似度计算(标题yl, 摘要xl, batch_size,可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s

        if max0_avg1_min2_sum3_t4_a5==0:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy,标题_摘要cos矩阵yx.T]).max(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==1:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy,标题_摘要cos矩阵yx.T]).mean(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==2:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx.T]).min(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==3:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx.T]).sum(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==4:
            论文cos矩阵xy = 标题_摘要cos矩阵xy
        else:
            论文cos矩阵xy = 标题_摘要cos矩阵yx.T
        return 论文cos矩阵xy,all_新词个数, all_新词s, all_加入新词s

    def 保存模型(self,address,save_step=False,max_to_keep=5):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        新建了Saver=False
        if address in self._保存模型地址_saver表:
            saver=self._保存模型地址_saver表[address]
        else:
            with self._sess.graph.as_default():
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                新建了Saver=True
        global_step = self._模型参数d['haveTrainingSteps'] if save_step else None
        saver.save(self._sess, address, global_step=global_step, write_meta_graph=True)
        global_step = '-'+str(global_step) if global_step else ''
        with open((address+global_step+'.parms').encode('utf-8'),'w',encoding='utf-8') as w: w.write(str(self._模型参数d))
        with open((address+global_step+'.w_num_map').encode('utf-8'),'wb') as w: w.write(pickle.dumps(self._词_序号d))
        return 新建了Saver

    def get_parms(self):
        return self._模型参数d

    def close(self):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        self._sess.close()
        if self._可视化w:
            self._可视化w.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

class InsuranceQA数据集:
    def __init__(self,数据目录,小写=False,使用还原词=False,相似概率矩阵地址=None):
        self._相似概率矩阵地址 = 相似概率矩阵地址
        vocabulary地址 = os.path.join(数据目录, 'vocabulary')
        answers地址 = os.path.join(数据目录, 'answers')
        dev地址 = os.path.join(数据目录, 'dev')
        test1地址 = os.path.join(数据目录, 'test1')
        test2地址 = os.path.join(数据目录, 'test2')
        train地址 = os.path.join(数据目录, 'train')
        with open(vocabulary地址.encode('utf-8'), 'rb') as r:
            self._序号_词表d = pickle.load(r) # {序号:'词',..}
        with open(answers地址.encode('utf-8'),'rb') as r:
            self._答案序号_句子d = pickle.load(r)

        with open(dev地址.encode('utf-8'),'rb') as r:
            self._dev问_好_坏l = pickle.load(r) # [{'question':[词号,..],'good':[答案号,..],'bad':[答案号,..]},..]
        with open(test1地址.encode('utf-8'),'rb') as r:
            self._test1问_好_坏l = pickle.load(r) # [{'question':[词号,..],'good':[答案号,..],'bad':[答案号,..]},..]
        with open(test2地址.encode('utf-8'),'rb') as r:
            self._test2问_好_坏l = pickle.load(r) # [{'question':[词号,..],'good':[答案号,..],'bad':[答案号,..]},..]
        with open(train地址.encode('utf-8'),'rb') as r:
            self._train问_答l = pickle.load(r) # [{'question':[词号,..],'answers':[答案号,..]},..]

        self._小写 = 小写
        self._所有答案句子 = []
        self._序号_所有答案句子序号 = {}
        for 序号,句子l in self._答案序号_句子d.items():
            self._所有答案句子.append(句子l)
            self._序号_所有答案句子序号[序号] = len(self._序号_所有答案句子序号)
        self._训练集QA相似概率矩阵 = []
        if 使用还原词:
            self.__数据集还原词处理()
        self._dev测试集l = self._转化测试集(self._dev问_好_坏l)
        self._test1测试集l = self._转化测试集(self._test1问_好_坏l)
        self._test2测试集l = self._转化测试集(self._test2问_好_坏l)

    def __数据集还原词处理(self):
        for 序号 in self._答案序号_句子d.keys():
            self._答案序号_句子d[序号] = self._句子_词序号还原(self._答案序号_句子d[序号])
        for d in self._dev问_好_坏l:
            d['question'] = self._句子_词序号还原(d['question'])
        for d in self._test1问_好_坏l:
            d['question'] = self._句子_词序号还原(d['question'])
        for d in self._test2问_好_坏l:
            d['question'] = self._句子_词序号还原(d['question'])
        for d in self._train问_答l:
            d['question'] = self._句子_词序号还原(d['question'])
        self._所有答案句子 = [self._句子_词序号还原(i) for i in self._所有答案句子]

    def _计算训练集问题答案TF_IDF概率距离(self, 相似度放大指数=1.):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        if self._相似概率矩阵地址:
            try:
                with open(self._相似概率矩阵地址.encode('utf-8'),'rb') as r:
                    l = pickle.load(r)
                print('直接读取')
                return l
            except:...

        startTime=time.time()
        文本l = []
        for d in self._train问_答l:
            文本l.append(' '.join(map(str, d['question'])))
        for 句子 in self._所有答案句子:
            文本l.append(' '.join(map(str, 句子)))
        tfidf = TfidfVectorizer(token_pattern="\S+",stop_words='english')
        re = tfidf.fit_transform(文本l)
        文本_相似概率矩阵l = cosine_similarity(re[:len(self._train问_答l)],re[len(self._train问_答l):], dense_output=False)
        最小值 = min(文本_相似概率矩阵l.data)
        # <class 'numpy.matrixlib.defmatrix.matrix'>, matrix 比 array 广播更直观
        文本_相似概率矩阵l = 文本_相似概率矩阵l.todense()
        if 相似度放大指数 != 1:
            文本_相似概率矩阵l = np.power(文本_相似概率矩阵l,相似度放大指数)
            最小值 = 最小值 ** 相似度放大指数
        文本_相似概率矩阵l += 最小值/len(self._所有答案句子)

        for i,d in enumerate(self._train问_答l):
            for 序号 in d['answers']:
                j = self._序号_所有答案句子序号[序号]
                文本_相似概率矩阵l[i,j] = 0.
        文本_相似概率矩阵l = 文本_相似概率矩阵l / 文本_相似概率矩阵l.sum(1)
        l = 文本_相似概率矩阵l.tolist()
        print('%.2fm'%((time.time()-startTime)/60))

        if self._相似概率矩阵地址:
            二进制 = pickle.dumps(l)
            缓存 = 10**4
            with open(self._相似概率矩阵地址.encode('utf-8'),'wb') as w:
                for i in tqdm(range(0,len(二进制),缓存),'写入相似度概率矩阵'):
                    w.write(二进制[i:i+缓存])
        return l
    def _句子_词序号还原(self,句子l):
        句子l_词 = []
        for 词序号 in 句子l:
            词 = self._序号_词表d[词序号]
            if self._小写:
                词 = 词.lower()
            句子l_词.append(词)
        return 句子l_词

    def _转化测试集(self,问_好_坏l):
        测试集l = []
        for one in 问_好_坏l:
            问题 = one['question']
            good答案l = [self._答案序号_句子d[i] for i in one['good']]
            bad答案l = [self._答案序号_句子d[i] for i in one['bad']]
            测试集l.append([问题, good答案l, bad答案l])
        return 测试集l

    def getTrainSet(self,使用tfidf筛选=0., tfidf相似度放大指数=1.):
        if 使用tfidf筛选 and len(self._训练集QA相似概率矩阵)!=len(self._train问_答l):
            self._训练集QA相似概率矩阵 = self._计算训练集问题答案TF_IDF概率距离(tfidf相似度放大指数)
        问题p, 问题n, 答案p, 答案n = [], [], [], []
        for i,一对 in enumerate(self._train问_答l):
            good答案l = [self._答案序号_句子d[i] for i in 一对['answers']]
            问题l = [一对['question']] * len(good答案l)
            if random.random() < 使用tfidf筛选:
                weights = self._训练集QA相似概率矩阵[i]
                bad答案l = []
                for j in range(len(good答案l)):
                    bad答案l.append(self._所有答案句子[np.random.choice(len(self._所有答案句子), p=weights)])
            else:
                bad答案l = random.sample(self._所有答案句子, len(good答案l))

            问题p += 问题l
            问题n += 问题l
            答案p += good答案l
            答案n += bad答案l
        return 问题p, 问题n, 答案p, 答案n

    def getTestSet(self,testSet = 'all'):
        if testSet == 'dev':
            return self._dev测试集l
        elif testSet == 'test1':
            return self._test1测试集l
        elif testSet == 'test2':
            return self._test2测试集l
        else:
            return {
                'dev': self._dev测试集l,
                'test1': self._test1测试集l,
                'test2': self._test2测试集l
            }

    def getWordEmbedding(self,词向量地址,前多少个):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        词_向量l = []
        vector = []
        with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8') as r:
            for line in r:
                line = line.strip().split(' ')
                if len(line) < 3:
                    continue
                word = line[0]
                vector = [float(i) for i in line[1:]]
                词_向量l.append([word, vector])
                if 前多少个<=len(词_向量l):
                    break
            维度 = len(vector)
        输出 = {
            'vec': 词_向量l,
            'embedding_dim': 维度,
            'vec_num': len(词_向量l),
        }
        print('vec_num:%d'%len(词_向量l))
        return 输出

class RAP数据集:
    def __init__(self,审稿人论文地址,稿件论文地址,训练集论文含稿件=True,小写=True,相似概率矩阵地址=None,论文编号位置=2):
        self._论文小写=小写
        self._训练集标题l = []
        self._训练集摘要l = []
        self._训练集论文l = []
        self._训练集论文编号s = set()
        self._训练集论文_相似概率矩阵l = []
        self._相似概率矩阵地址 = 相似概率矩阵地址

        self._审稿人论文信息l = self._get论文信息(审稿人论文地址)
        self._审稿人标题l, self._审稿人摘要l = self._分离论文TA(self._审稿人论文信息l)
        self._审稿人论文编号l = [i[2] for i in self._审稿人论文信息l]
        self._审稿人编号l = [i[3] for i in self._审稿人论文信息l]

        self._稿件论文信息l = self._get论文信息(稿件论文地址)
        self._稿件标题l, self._稿件摘要l = self._分离论文TA(self._稿件论文信息l)
        self._稿件编号l = [i[2] for i in self._稿件论文信息l]

        self.增加训练集语料(self._审稿人论文信息l,论文编号位置=论文编号位置)
        if 训练集论文含稿件:
            self.增加训练集语料(self._稿件论文信息l,论文编号位置=论文编号位置)
        print('审稿人论文数:%d, 稿件论文数:%d'%(len(self._审稿人论文信息l),len(self._稿件论文信息l)))

    def _get论文信息(self,论文地址):
        论文信息l = []
        with open(论文地址.encode('utf-8'),'r',encoding='utf-8') as r:
            for line in r:
                if self._论文小写:
                    line = line.lower()
                line = line.strip().split('\t')
                if len(line)<2:
                    continue
                标题, 摘要 = line[:2]
                标题l = 标题.split(' ')
                摘要l = 摘要.split(' ')
                论文信息l.append([标题l,摘要l]+line[2:])
        return 论文信息l

    def _分离论文TA(self,论文信息l):
        标题l, 摘要l = [], []
        for x in 论文信息l:
            标题, 摘要 = x[:2]
            标题l.append(标题)
            摘要l.append(摘要)
        return 标题l, 摘要l

    def _计算文本TF_IDF概率距离(self,文本_词矩阵l, 相似度放大指数=1.):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '(%d论文)...'%len(文本_词矩阵l),end='')
        sys.stdout.flush()
        if self._相似概率矩阵地址:
            try:
                with open(self._相似概率矩阵地址.encode('utf-8'),'rb') as r:
                    l = pickle.load(r)
                print('直接读取')
                return l
            except:...

        startTime=time.time()
        tfidf = TfidfVectorizer(token_pattern="\S+",stop_words='english')
        re = tfidf.fit_transform([' '.join(i) for i in 文本_词矩阵l])
        文本_相似概率矩阵l = cosine_similarity(re, dense_output=False)
        最小值 = min(文本_相似概率矩阵l.data)
        # <class 'numpy.matrixlib.defmatrix.matrix'>, matrix 比 array 广播更直观
        文本_相似概率矩阵l = 文本_相似概率矩阵l.todense()
        if 相似度放大指数 != 1:
            文本_相似概率矩阵l = np.power(文本_相似概率矩阵l,相似度放大指数)
            最小值 = 最小值 ** 相似度放大指数
        文本_相似概率矩阵l += 最小值 / len(文本_词矩阵l)
        for i in range(len(文本_相似概率矩阵l)): 文本_相似概率矩阵l[i, i] = 0.
        文本_相似概率矩阵l = 文本_相似概率矩阵l / 文本_相似概率矩阵l.sum(1)
        l = 文本_相似概率矩阵l.tolist()
        print('%.2fm'%((time.time()-startTime)/60))

        if self._相似概率矩阵地址:
            二进制 = pickle.dumps(l)
            缓存 = 10**4
            with open(self._相似概率矩阵地址.encode('utf-8'),'wb') as w:
                for i in tqdm(range(0,len(二进制),缓存),'写入相似度概率矩阵'):
                    w.write(二进制[i:i+缓存])
        return l

    def 增加训练集语料(self,论文地址或信息,论文编号位置 = None):
        if isinstance(论文地址或信息,str):
            论文信息l = self._get论文信息(论文地址或信息)
        else:
            论文信息l = 论文地址或信息
        for x in 论文信息l:
            标题l, 摘要l = x[:2]
            if 论文编号位置:
                if x[论文编号位置] in self._训练集论文编号s:
                    continue
                self._训练集论文编号s.add(x[论文编号位置])
            self._训练集标题l.append(标题l)
            self._训练集摘要l.append(摘要l)
            self._训练集论文l.append(标题l+摘要l)

    def getTrainSet(self,使用tfidf筛选=0., tfidf相似度放大指数=1.):
        if 使用tfidf筛选 and len(self._训练集论文l)!=len(self._训练集论文_相似概率矩阵l):
            self._训练集论文_相似概率矩阵l = self._计算文本TF_IDF概率距离(self._训练集论文l, tfidf相似度放大指数)
        标题n, 摘要n = [], []
        for i,(标题,摘要) in enumerate(zip(self._训练集标题l,self._训练集摘要l)):
            if random.random() < 使用tfidf筛选:
                weights = self._训练集论文_相似概率矩阵l[i]
            else:
                weights = None
            if random.random()<0.5:
                标题n.append(self._训练集标题l[np.random.choice(len(self._训练集标题l), p=weights)])
                摘要n.append(摘要)
            else:
                标题n.append(标题)
                摘要n.append(self._训练集摘要l[np.random.choice(len(self._训练集摘要l), p=weights)])
        return self._训练集标题l, 标题n, self._训练集摘要l, 摘要n

    def get实战数据(self):
        return [self._稿件标题l, self._稿件摘要l], [self._审稿人标题l, self._审稿人摘要l]

    def getWordEmbedding(self,词向量地址,前多少个):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        词_向量l = []
        vector = []
        with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8') as r:
            for line in r:
                line = line.strip().split(' ')
                if len(line) < 3:
                    continue
                word = line[0]
                vector = [float(i) for i in line[1:]]
                词_向量l.append([word, vector])
                if 前多少个<=len(词_向量l):
                    break
            维度 = len(vector)
        输出 = {
            'vec': 词_向量l,
            'embedding_dim': 维度,
            'vec_num': len(词_向量l),
        }
        print('vec_num:%d'%len(词_向量l))
        return 输出

    def 获得预测标签(self,稿件_审稿人论文相似度矩阵,求和算法=lambda x:sum(x[:50]),得分矩阵输出地址=None):
        审稿人_论文8相似度l = []
        for 审稿人论文相似度l in 稿件_审稿人论文相似度矩阵:
            审稿人_论文8相似度d = {}
            for i, 相似度 in enumerate(审稿人论文相似度l):
                审稿人论文编号 = self._审稿人论文编号l[i]
                审稿人编号 = self._审稿人编号l[i]
                if 审稿人编号 in 审稿人_论文8相似度d: 审稿人_论文8相似度d[审稿人编号][审稿人论文编号] = 相似度
                else: 审稿人_论文8相似度d[审稿人编号] = {审稿人论文编号: 相似度}
            审稿人_论文8相似度l.append(审稿人_论文8相似度d)

        稿件编号_专家编号_得分l = []
        稿件编号_专家编号d = {}
        for 稿件编号, 审稿人_论文8相似度d in zip(self._稿件编号l, 审稿人_论文8相似度l):
            稿件编号_专家编号_得分l.append([稿件编号])
            审稿人编号_相似度l = []
            for 审稿人编号, 论文_相似度d in 审稿人_论文8相似度d.items():
                相似度 = 求和算法(sorted([i for i in 论文_相似度d.values()],reverse=True))
                稿件编号_专家编号_得分l[-1] += [审稿人编号, 相似度]
                审稿人编号_相似度l.append([审稿人编号, 相似度])
            稿件编号_专家编号d[稿件编号] = [i for i,_ in sorted(审稿人编号_相似度l,key=lambda t:t[1],reverse=True)]
        if 得分矩阵输出地址:
            with open(得分矩阵输出地址.encode('utf-8'), 'w', encoding='utf-8') as w:
                for i in 稿件编号_专家编号_得分l:
                    w.write('\t'.join([str(j) for j in i]))
                    w.write('\n')
        return 稿件编号_专家编号d

class RAP评估:
    def __init__(self,标签地址):
        self._稿件编号_实际专家d = self._载入标准评价(标签地址)

    def _载入标准评价(self,标准评价地址):
        稿件编号_实际专家d={}
        with open(标准评价地址.encode('utf-8'),'r',encoding='utf-8') as r:
            for line in r:
                line=line.strip().split('\t')
                稿件编号,专家编号l = os.path.splitext(line[0])[0],line[1:]
                稿件编号_实际专家d[稿件编号]=专家编号l
        return 稿件编号_实际专家d

    def 评估(self,预测标签,topN=20,简化=True,输出地址=None):
        预测向量组l, 标签向量组l = [], []
        稿件编号l = []
        for 稿件编号, 专家l in self._稿件编号_实际专家d.items():
            标签向量组l.append(专家l)
            预测向量组l.append(预测标签[稿件编号])
            稿件编号l.append(稿件编号)

        batch_top_FP, batch_top_FN, batch_top_TP = [], [], []
        MAP_l, NDCG_l, bpref_l = [], [], []
        for 预测向量组, 标签向量组 in zip(预测向量组l, 标签向量组l):
            top_FP, top_FN, top_TP = [], [], []
            if 简化:
                top遍历 = [topN]
            else:
                top遍历 = range(1,topN+1)

            标签向量组s = set(标签向量组)
            正例数 = len(标签向量组s)
            for j in  top遍历:
                预测正确数 = len(set(预测向量组[:j])&标签向量组s)
                top_FP.append(j-预测正确数)
                top_FN.append(正例数-预测正确数)
                top_TP.append(预测正确数)
                MAP_l.append(self.MAP_相关文档数为N(预测向量组l, 标签向量组l, j))
                NDCG_l.append(self.NDCG_无序(预测向量组l, 标签向量组l, j))
                bpref_l.append(self.Bpref_相关文档数为N(预测向量组l, 标签向量组l, j))
            batch_top_FP.append(top_FP)
            batch_top_FN.append(top_FN)
            batch_top_TP.append(top_TP)

        batch_top_FP=np.array(batch_top_FP)
        batch_top_FN=np.array(batch_top_FN)
        batch_top_TP=np.array(batch_top_TP)
        batch_top_P = batch_top_TP/(batch_top_TP+batch_top_FP)
        batch_top_R = batch_top_TP/(batch_top_TP+batch_top_FN)
        top_macroP = batch_top_P.mean(axis=0)
        top_macroR = batch_top_R.mean(axis=0)
        top_macroF1 = 2*top_macroP*top_macroR/(top_macroP+top_macroR)

        if 简化:
            top_macroP = float(top_macroP[0])
            top_macroR = float(top_macroR[0])
            top_macroF1 = float(top_macroF1[0])
            MAP = MAP_l[0]
            NDCG = NDCG_l[0]
            bpref = bpref_l[0]
        else:
            top_macroP = top_macroP.tolist()
            top_macroR = top_macroR.tolist()
            top_macroF1 = top_macroF1.tolist()
            MAP = MAP_l
            NDCG = NDCG_l
            bpref = bpref_l
        输出 = {
            'macro-P': top_macroP,
            'macro-R': top_macroR,
            'macro-F1': top_macroF1,
            'MAP': MAP,
            'NDCG': NDCG,
            'bpref': bpref,
        }
        if 输出地址:
            with open(输出地址.encode('utf-8'),'w',encoding='utf-8') as w:
                w.write('all: '+str(输出)+'\n')
                w.write('manuscript\tP\tR\n')
                for 稿件编号,top_P,top_R in zip(稿件编号l,batch_top_P,batch_top_R):
                    top_P, top_R = top_P.tolist(),top_R.tolist()
                    w.write(稿件编号+'\t')
                    if 简化:
                        w.write('%.4f\t%.4f\n'%(top_P[-1],top_R[-1]))
                    else:
                        w.write('%s\t%s\n'%(str(top_P),str(top_R)))
        return 输出

    @staticmethod
    def MAP_相关文档数为N(预测向量组l, 标签向量组l, N):
        AP = []
        相关文档数 = N
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            准确个数 = 0
            ap = 0
            for j in range(相关文档数):
                if 预测向量l[j] in 标签向量s:
                    准确个数 += 1
                    ap += 准确个数 / (j + 1)
            AP.append(ap / 相关文档数)
        return sum(AP) / len(AP)

    @staticmethod
    def NDCG_无序(预测向量组l, 标签向量组l, N):
        ndcg = 0
        idcg = sum(1 / math.log2(i + 2) for i in range(N))
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            for j in range(len(预测向量l)):
                if 预测向量l[j] in 标签向量s:
                    ndcg += 1 / math.log2(j + 2)
        return ndcg / idcg / len(预测向量组l)

    @staticmethod
    def Bpref_相关文档数为N(预测向量组l, 标签向量组l, N):
        bpref = 0
        相关文档数 = N
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            不准确个数 = 0
            for j in range(相关文档数):
                if 预测向量l[j] not in 标签向量s:
                    不准确个数 += 1
                bpref += (1 - 不准确个数 / 相关文档数)
        return bpref / 相关文档数 / len(预测向量组l)


def 运行_InsuranceQA():
    ap = ''

    batchSize = 122
    epochs = 100000
    多少批次记录一次 = 10
    记录第一批次元数据 = True
    获取前多少个词向量 = 100000
    重复训练多少上一轮训练错误的数据 = int(batchSize*0.)
    使用还原词 = True
    BERT句词向量目录 = '/home/tansc/python/insuranceqa-pytorch-master/insurance_qa_python/BERT_embedding.pkl'

    使用tfidf筛选 = 0.5
    tfidf相似度放大指数 = 1
    多少轮使用一次tfidf筛选 = 1

    多少轮测试一次模型 = 2
    batch_size_测试集 = 500
    目前最好结果 = [0, 0]

    模型地址 = 'D:\data\code\python\GPU-31-11/test/model/InsuranceQA_model4'
    数据集 = InsuranceQA数据集('D:\data\code\python\GPU-31-11/insuranceqa-pytorch-master/insurance_qa_python',小写=True,使用还原词=使用还原词)
    词_向量l = 数据集.getWordEmbedding(词向量地址='D:\data\code\python\GPU-31-11\insuranceqa-pytorch-master\insurance_qa_python/train_answers_number_vectors.txt',
    # 模型地址 = '/home/tansc/python/test/model/InsuranceQA_model4'
    # 数据集 = InsuranceQA数据集('/home/tansc/python/insuranceqa-pytorch-master/insurance_qa_python',小写=True,使用还原词=使用还原词)
    # 词_向量l = 数据集.getWordEmbedding(词向量地址='/home/tansc/c/word2vec/train_answers_number_vectors.txt',
                                 前多少个=获取前多少个词向量)['vec']

    训练错误的数据l = []
    with 标题摘要_tf模型(模型参数d, 初始词向量l=词_向量l, 可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
    # with 标题摘要_tf模型(模型参数d,可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
    # with 标题摘要_tf模型(模型参数=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
        总批次 = model.get_parms()['haveTrainingSteps']
        记录过程 = False
        for epoch in range(epochs):
            if epoch%多少轮使用一次tfidf筛选 == 0:
                问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet(使用tfidf筛选=使用tfidf筛选, tfidf相似度放大指数=tfidf相似度放大指数)
            else:
                问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet()
            多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s = model.预_编号与填充_批量([问题p, 问题n, 答案p, 答案n],[1,1,0,0])
            问题p, 问题n, 答案p, 答案n = 多_句_词矩阵l
            问题p长度l, 问题n长度l, 答案p长度l, 答案n长度l = 多_长度l
            print('训练集大小:%d, 新词数:%d, 新词s:%d, 加入新词s:%d' % (len(问题p), all新词数, len(all新词s), len(all加入新词s)))

            loss_all, acc_all = [], []
            for batch in tqdm(range(0,len(问题p),batchSize)):
                if 总批次%多少批次记录一次==0:
                    记录过程 = True
                输出 = model.训练(title_p=问题p[batch:batch + batchSize], abstract_p=答案p[batch:batch + batchSize],
                                 title_n=问题n[batch:batch + batchSize], abstract_n=答案n[batch:batch + batchSize],
                                 title_len_p=问题p长度l[batch:batch + batchSize],
                                 abstract_len_p=答案p长度l[batch:batch + batchSize],
                                 title_len_n=问题n长度l[batch:batch + batchSize],
                                 abstract_len_n=答案n长度l[batch:batch + batchSize],
                                 记录过程=记录过程,
                                 记录元数据=记录第一批次元数据,
                                 合并之前训练错误的数据=[i[:重复训练多少上一轮训练错误的数据] for i in 训练错误的数据l])
                # 训练错误的数据l = 输出['训练错误的数据']
                训练错误的数据l = 输出['损失函数大于0的数据']
                总批次 += 1
                记录过程 = False
                记录第一批次元数据 = False
                loss_all.append(输出['损失函数值'])
                acc_all.append(输出['精确度'])
            print('总批次:%d, epoch:%d, loss:%f, acc:%f'%
                  (总批次, epoch+1, sum(loss_all)/len(loss_all), sum(acc_all)/len(acc_all)))
            print()
            if epoch%多少轮测试一次模型==0:
                测试集l = 数据集.getTestSet('dev')
                print('验证集:dev, 目前最好结果:%.4f(%d-epochs)'%(目前最好结果[0],目前最好结果[1]))
                top指标, all_新词个数, all_新词s, all_加入新词s = model.测试(测试集l, batch_size=batch_size_测试集)
                top_1 = top指标['macro_P'][0]
                if top_1 >= 目前最好结果[0]:
                    目前最好结果[0] = top_1
                    目前最好结果[1] = epoch+1
                    model.保存模型(模型地址)
                    print('保存了一次模型: %d step' % 总批次)
                print('top@1:%.4f(%d-epochs), 新词数:%d, 新词s:%d, 加入新词s:%d' % (top_1, epoch+1, all_新词个数, len(all_新词s), len(all_加入新词s)))
                print()

def 运行_InsuranceQA测试():
    ap = ''

    topN=10
    batch_size = 500
    使用还原词 = True
    模型地址 = 'D:\data\code\python\GPU-31-11/test/model/InsuranceQA_model4'
    数据集 = InsuranceQA数据集('D:\data\code\python\GPU-31-11/insuranceqa-pytorch-master/insurance_qa_python',小写=True,使用还原词=使用还原词)
    # 模型地址 = '/home/tansc/python/test/model/InsuranceQA_model11'
    # 数据集 = InsuranceQA数据集('/home/tansc/python/insuranceqa-pytorch-master/insurance_qa_python',小写=True,使用还原词=使用还原词)
    BERT句词向量目录 = '/home/tansc/python/insuranceqa-pytorch-master/insurance_qa_python/BERT_embedding.pkl'

    with 标题摘要_tf模型(模型参数=模型地址, 取消可视化=True, BERT句词向量目录=BERT句词向量目录) as model:
        for 测试集名, 测试集l in 数据集.getTestSet().items():
            print('测试集:%s' % 测试集名)
            top指标, all_新词个数, all_新词s, all_加入新词s = model.测试(测试集l, topN=topN, batch_size=batch_size)
            print('新词数:%d, 新词s:%d, 加入新词s:%d' % (all_新词个数, len(all_新词s), len(all_加入新词s)))
            print(''.join(['\ttop@%d'%(i+1) for i in range(topN)]))
            for 指标名,值l in top指标.items():
                print(指标名+'\t'+'\t'.join(['%.4f'%i for i in 值l]))
            print()

def 运行_RAP():
    ap = '/home/tansc/python/paper/RAP/SPM-RA/'
    # ap = '/home/tansc/python/test/data/paper/10/'

    batchSize = 122
    epochs = 100000
    多少批次记录一次 = 10
    记录第一批次元数据 = True
    获取前多少个词向量 = 150000
    重复训练多少上一轮训练错误的数据 = int(batchSize*0.)
    BERT句词向量目录 = ap+'aa_BERT句向量.pkl'
    # 词向量地址 = '/home/tansc/c/word2vec/elsevier_all_vectors.txt'
    # 词向量地址 = '/home/tansc/c/word2vec/arxivDoubleCorpus_vectors.txt'

    使用tfidf筛选 = 0.5
    tfidf相似度放大指数 = 1
    多少轮使用一次tfidf筛选 = 1

    多少轮测试一次模型 = 1
    batch_size_测试集 = 500
    目前最好结果 = [0, 0]
    max0_avg1_min2_sum3_t4_a5 = 3
    # 评估结果输出地址 = ap+'aa_RAP结果.txt'
    评估结果输出地址 = None
    求和算法 = lambda x: sum(x[:50])
    topN = 20

    # 模型地址 = ap+'elsevier3/SPM'
    模型地址 = ap+'arxiv7/SPM'
    数据集 = RAP数据集(审稿人论文地址=ap+'ac_作者论文.txt',
                 稿件论文地址=ap+'ac_稿件论文.txt',
                 训练集论文含稿件=True,
                 小写=True,
                 相似概率矩阵地址=ap+'aa_相似概率矩阵.pkl',
                 论文编号位置=None)
    评估 = RAP评估(标签地址=ap+'文档-标准作者排名.txt')

    # 词_向量l = 数据集.getWordEmbedding(词向量地址=词向量地址, 前多少个=获取前多少个词向量)['vec']
    训练错误的数据l = []
    # with 标题摘要_tf模型(模型参数d,初始词向量l=词_向量l,可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
    # with 标题摘要_tf模型(模型参数d, 可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
    with 标题摘要_tf模型(模型参数=模型地址, BERT句词向量目录=BERT句词向量目录) as model:
        总批次 = model.get_parms()['haveTrainingSteps']
        记录过程 = False
        for epoch in range(epochs):
            if epoch%多少轮使用一次tfidf筛选 == 0:
                问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet(使用tfidf筛选=使用tfidf筛选, tfidf相似度放大指数=tfidf相似度放大指数)
            else:
                问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet()
            多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s = model.预_编号与填充_批量([问题p, 问题n, 答案p, 答案n],[1,1,0,0])
            问题p, 问题n, 答案p, 答案n = 多_句_词矩阵l
            问题p长度l, 问题n长度l, 答案p长度l, 答案n长度l = 多_长度l
            print('训练集大小:%d, 新词数:%d, 新词s:%d, 加入新词s:%d' % (len(问题p), all新词数, len(all新词s), len(all加入新词s)))

            loss_all, acc_all = [], []
            for batch in tqdm(range(0, len(问题p), batchSize)):
                if 总批次 % 多少批次记录一次 == 0:
                    记录过程 = True
                输出 = model.训练(title_p=问题p[batch:batch + batchSize], abstract_p=答案p[batch:batch + batchSize],
                              title_n=问题n[batch:batch + batchSize], abstract_n=答案n[batch:batch + batchSize],
                              title_len_p=问题p长度l[batch:batch + batchSize],
                              abstract_len_p=答案p长度l[batch:batch + batchSize],
                              title_len_n=问题n长度l[batch:batch + batchSize],
                              abstract_len_n=答案n长度l[batch:batch + batchSize],
                              记录过程=记录过程,
                              记录元数据=记录第一批次元数据,
                              合并之前训练错误的数据=[i[:重复训练多少上一轮训练错误的数据] for i in 训练错误的数据l])
                # 训练错误的数据l = 输出['训练错误的数据']
                训练错误的数据l = 输出['损失函数大于0的数据']
                总批次 += 1
                记录过程 = False
                记录第一批次元数据 = False
                loss_all.append(输出['损失函数值'])
                acc_all.append(输出['精确度'])
            print('总批次:%d, epoch:%d, loss:%f, acc:%f' %
                  (总批次, epoch + 1, sum(loss_all) / len(loss_all), sum(acc_all) / len(acc_all)))
            print()
            if epoch%多少轮测试一次模型==0:
                稿件论文, 审稿人论文 = 数据集.get实战数据()
                print('目前最好结果P:%.4f(%d-epochs)'%(目前最好结果[0],目前最好结果[1]))
                论文cos矩阵xy, all_新词个数, all_新词s, all_加入新词s = model.论文相似度计算(
                                        论文xl=稿件论文,
                                        论文yl=审稿人论文,
                                        batch_size=batch_size_测试集,
                                        max0_avg1_min2_sum3_t4_a5=max0_avg1_min2_sum3_t4_a5)
                预测标签 = 数据集.获得预测标签(论文cos矩阵xy.tolist(),
                                  求和算法=求和算法,
                                  得分矩阵输出地址=None)
                输出 = 评估.评估(预测标签,topN=topN,简化=True,输出地址=评估结果输出地址)
                if 输出['macro-P'] >= 目前最好结果[0]:
                    目前最好结果[0] = 输出['macro-P']
                    目前最好结果[1] = epoch+1
                    model.保存模型(模型地址)
                    print('保存了一次模型: %d step' % 总批次)
                print('%d-epoch 新词数:%d, 新词s:%d, 加入新词s:%d' % (epoch+1,all_新词个数, len(all_新词s), len(all_加入新词s)))
                for 指标名, 值 in 输出.items():
                    print('%s: %.4f, '%(指标名,值),end='')
                print('\n')

def 运行_RAP测试():
    ap = ''

    batch_size = 500
    max0_avg1_min2_sum3_t4_a5 = 3
    评估结果输出地址 = ap + '/home/tansc/python/test/data/paper/10/result.txt'
    求和算法 = lambda x: sum(x[:50])
    topN = 20
    BERT句词向量目录 = '/home/tansc/python/test/data/paper/10/BERT_embedding'

    模型地址 = '/home/tansc/python/test/data/model/elsevier8'
    数据集 = RAP数据集(审稿人论文地址='/home/tansc/python/test/data/paper/10/author.txt',
                 稿件论文地址='/home/tansc/python/test/data/paper/10/manuscript.txt',
                 训练集论文含稿件=True,
                 小写=True,
                 论文编号位置=2)
    评估 = RAP评估(标签地址='/home/tansc/python/test/data/paper/10/label.txt')

    with 标题摘要_tf模型(模型参数=模型地址, 取消可视化=True, BERT句词向量目录=BERT句词向量目录) as model:
        稿件论文, 审稿人论文 = 数据集.get实战数据()
        论文cos矩阵xy, all_新词个数, all_新词s, all_加入新词s = model.论文相似度计算(
                                论文xl=稿件论文,
                                论文yl=审稿人论文,
                                batch_size=batch_size,
                                max0_avg1_min2_sum3_t4_a5=max0_avg1_min2_sum3_t4_a5)
        预测标签 = 数据集.获得预测标签(论文cos矩阵xy.tolist(),
                          求和算法=求和算法,
                          得分矩阵输出地址=None)
        输出 = 评估.评估(预测标签,topN=topN,简化=True,输出地址=评估结果输出地址)
        print('新词数:%d, 新词s:%d, 加入新词s:%d' % (all_新词个数, len(all_新词s), len(all_加入新词s)))
        for 指标名, 值 in 输出.items():
            print('%s: %.4f, '%(指标名,值),end='')


模型参数d={
    '显存占用比': 0.5,
    'title_maxlen': 200,
    'abstract_maxlen': 200,
    'embedding_dim':100,
    'margin': 0.1,
    '共享参数': True,

    'learning_rate': 0.3,
    '学习率衰减步数': 1000,
    '学习率衰减率': 0.7,
    '学习率最小值倍数': 100,
    'AdamOptimizer': 0.001,

    '词数上限': 150000,
    '词向量固定值初始化': None,
    '固定词向量': False,
    '词向量微调': False,
    '可加入新词': True,
    '词向量tanh': False,

    '使用LSTM': True,
    'biLSTM_各隐层数':[200],
    'biLSTM池化方法': 'tf.concat',
    'LSTM序列池化方法': 'tf.reduce_max',
    'LSTM_dropout': 1.,

    '使用CNN': True,
    'filter_sizes': [1,2,3,5],
    'num_filters': 1024,
    'CNN_dropout': 1.,
    'CNN输出层tanh': False,

    '使用BERT': False,
    'BERT_maxlen': 200,
    'BERT_embedding_dim': 1024,
    '使用[CLS]': True,
    '使用[SEP]': True,
    'BERT句向量存取加速': True,
}

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 运行_InsuranceQA()
    运行_RAP()
    # 运行_InsuranceQA测试()
    # 运行_RAP测试()

'''
使用BERT前打开服务,比如执行命令: bert-serving-start -model_dir /home/tansc/python/fork/uncased_L-24_H-1024_A-16 -gpu_memory_fraction=0.1 -max_seq_len=200 -max_batch_size=64 -num_worker=1 -pooling_strategy=NONE
'''