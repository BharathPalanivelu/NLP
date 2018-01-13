import os
import sys
import numpy as np


def read_data(filename):
    with open(filename) as file:
        train = file.readlines()

    sentences = []
    one_sentence = [[0, '**ROOT**', '_', '**ROOT**', '**ROOT**', '_', -1, '**ROOT**', '_', '_']]

    for line in train:
        if line != '\n':
            one_sentence.append(line.split('\t'))
        else:
            sentences.append(one_sentence)
            one_sentence = [[0, '**ROOT**', '_', '**ROOT**', '**ROOT**', '_', -1, '**ROOT**', '_', '_']]

    return sentences    
    

def train(train_data, lr=1, n_iter=10):
    # how many iterations over all traning data?
    for i in range(n_iter):
        # reading the whole training data
        sentences = read_data(train_data)
        # training, for each individual sentence
        for j in range(len(sentences)):
            sen = sentences.pop(0)
            
            # change index and head in each line from str to int
            for line in sen:
                line[0] = int(line[0])
                line[6] = int(line[6])
                line[1] = line[1].lower()
            
            # initialize restriction for right-arc
            restriction = None
            
            # initialize stack, buffer, and actions
            stack = [0]
            buffer = [word[0] for word in sen[1:]]
            actions = []
            
            ## get actions
            while len(stack) > 0 and len(buffer) > 0:
                if sen[ stack[-1] ][6] == buffer[0]:
                    action = 'L'
                    stack.pop(-1)
                elif sen[ buffer[0] ][6] == stack[-1]:
                    for b in buffer[1:]:
                        if sen[b][6] == buffer[0]:
                            restriction = True
                    if restriction:
                        action = 'S'
                        stack.append(buffer.pop(0))
                    else:
                        action = 'R'
                        buffer[0] = stack.pop(-1)
                    restriction = False # back to default
                else:
                    action = 'S'
                    stack.append(buffer.pop(0))
                actions.append(action)
                
            # configuration and update
            # re-initialize
            stack = [0]
            buffer = [word[0] for word in sen[1:]]
            
            
            ## predict
            delta = {'L_bias': 0., 'R_bias': 0., 'S_bias': 0.} # cached weights and bias
            c = 0 # num of wrong predictions
            for config in range(len(actions)):
                
                # get features
                # f1 - identity of word at top of the stack
                f1_L = 'L_stack_top_word=' + sen[ stack[-1] ][1]
                f1_R = 'R_stack_top_word=' + sen[ stack[-1] ][1]
                f1_S = 'S_stack_top_word=' + sen[ stack[-1] ][1]
                # f2 - identity of word at head of buffer
                f2_L = 'L_buffer_head_word=' + sen[ buffer[0] ][1]
                f2_R = 'R_buffer_head_word=' + sen[ buffer[0] ][1]
                f2_S = 'S_buffer_head_word=' + sen[ buffer[0] ][1]
                # f3 - coarse POS (field 4) of word at top of the stack
                f3_L = 'L_stack_top_POS=' + sen[ stack[-1] ][3]
                f3_R = 'R_stack_top_POS=' + sen[ stack[-1] ][3]
                f3_S = 'S_stack_top_POS=' + sen[ stack[-1] ][3]
                # f4 - coarse POS (field 4) of word at head of buffer
                f4_L = 'L_buffer_head_POS=' + sen[ buffer[0] ][3]
                f4_R = 'R_buffer_head_POS=' + sen[ buffer[0] ][3]
                f4_S = 'S_buffer_head_POS=' + sen[ buffer[0] ][3]
                # f5 - pair of words at top of stack and head of buffer
                f5_L = 'L_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
                f5_R = 'R_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
                f5_S = 'S_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
                # f6 - pair of coarse POS (field 4) at top of stack and head of buffer
                f6_L = 'L_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
                f6_R = 'R_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
                f6_S = 'S_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
                # feature list
                f_list = [f1_L, f1_R, f1_S, f2_L, f2_R, f2_S,
                          f3_L, f3_R, f3_S, f4_L, f4_R, f4_S, 
                          f5_L, f5_R, f5_S, f6_L, f6_R, f6_S]
                
                # initialize weights
                for f in f_list:
                    if f not in features:
                        features[f] = 0.
                    if f not in delta:
                        delta[f] = 0.
                
                L_weights = np.array([features[L] for L in f_list if L[0] == 'L'])
                R_weights = np.array([features[L] for L in f_list if L[0] == 'R'])
                S_weights = np.array([features[L] for L in f_list if L[0] == 'S'])
                
                # dot product
                L_dot = np.dot(np.ones((1,6)), L_weights) + features['L_bias']
                R_dot = np.dot(np.ones((1,6)), R_weights) + features['R_bias']
                S_dot = np.dot(np.ones((1,6)), S_weights) + features['S_bias']
                
                pred_action = np.argmax([L_dot, R_dot, S_dot])
                pred_action = ['L', 'R', 'S'][pred_action]
                
                # check and update
                if pred_action != actions[config]:
                    for feat in f_list:
                        # increase gold features
                        if feat[0] == actions[config]:
                            delta[feat] += lr * 1
                            # features[feat] += lr * 1
                        # decrease pred features
                        elif feat[0] == pred_action:
                            delta[feat] -= lr * 1
                            # features[feat] -= lr * 1
                    # update bias
                    # features[actions[config] + '_bias'] += lr * 1
                    # features[pred_action + '_bias'] -= lr * 1
                    delta[actions[config] + '_bias'] += lr * 1
                    delta[pred_action + '_bias'] -= lr * 1
                    # update c
                    c += 1
                
                # move to next configuration
                if actions[config] == 'L':
                    stack.pop(-1)
                elif actions[config] == 'R':
                    buffer[0] = stack.pop(-1)
                else:
                    stack.append(buffer.pop(0))
            
            # update actual features
            # method 1 - averaged by length of actions
            for delta_feat in delta:
                features[delta_feat] += 1/len(actions) * delta[delta_feat]
            # method 2 - averaged by number of mistakes
#            for delta_feat in delta:
#                if c == 0:
#                    features[delta_feat] += delta[delta_feat]
#                else:
#                    features[delta_feat] += 1/c * delta[delta_feat]


def predict(test_data, features):
    # important not to change initial features
    feat_copy = features.copy()
    results = []
    sentences = read_data(test_data)
    
    for sen in sentences:
        for line in sen:
            line[0] = int(line[0])
            line[1] = line[1].lower()
            
        ## initialize
        stack = [0]
        buffer = [word[0] for word in sen[1:]]
        heads = [-1] * len(sen)
        
        ## go through a sentence
        # features
        while len(stack) > 0 and len(buffer) > 0:
            # f1 - identity of word at top of the stack
            f1_L = 'L_stack_top_word=' + sen[ stack[-1] ][1]
            f1_R = 'R_stack_top_word=' + sen[ stack[-1] ][1]
            f1_S = 'S_stack_top_word=' + sen[ stack[-1] ][1]
            # f2 - identity of word at head of buffer
            f2_L = 'L_buffer_head_word=' + sen[ buffer[0] ][1]
            f2_R = 'R_buffer_head_word=' + sen[ buffer[0] ][1]
            f2_S = 'S_buffer_head_word=' + sen[ buffer[0] ][1]
            # f3 - coarse POS (field 4) of word at top of the stack
            f3_L = 'L_stack_top_POS=' + sen[ stack[-1] ][3]
            f3_R = 'R_stack_top_POS=' + sen[ stack[-1] ][3]
            f3_S = 'S_stack_top_POS=' + sen[ stack[-1] ][3]
            # f4 - coarse POS (field 4) of word at head of buffer
            f4_L = 'L_buffer_head_POS=' + sen[ buffer[0] ][3]
            f4_R = 'R_buffer_head_POS=' + sen[ buffer[0] ][3]
            f4_S = 'S_buffer_head_POS=' + sen[ buffer[0] ][3]
            # f5 - pair of words at top of stack and head of buffer
            f5_L = 'L_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
            f5_R = 'R_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
            f5_S = 'S_pair_word=' + sen[ stack[-1] ][1] + '_' + sen[ buffer[0] ][1]
            # f6 - pair of coarse POS (field 4) at top of stack and head of buffer
            f6_L = 'L_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
            f6_R = 'R_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
            f6_S = 'S_pair_POS=' + sen[ stack[-1] ][3] + '_' + sen[ buffer[0] ][3]
            # feature list
            f_list = [f1_L, f1_R, f1_S, f2_L, f2_R, f2_S,
                      f3_L, f3_R, f3_S, f4_L, f4_R, f4_S, 
                      f5_L, f5_R, f5_S, f6_L, f6_R, f6_S]
            
            # initialize weights
            for f in f_list:
                if f not in feat_copy:
                    feat_copy[f] = 0.
            
            L_weights = np.array([feat_copy[L] for L in f_list if L[0] == 'L'])
            R_weights = np.array([feat_copy[L] for L in f_list if L[0] == 'R'])
            S_weights = np.array([feat_copy[L] for L in f_list if L[0] == 'S'])
            
            # dot product
            L_dot = np.dot(np.ones((1,6)), L_weights) + feat_copy['L_bias']
            R_dot = np.dot(np.ones((1,6)), R_weights) + feat_copy['R_bias']
            S_dot = np.dot(np.ones((1,6)), S_weights) + feat_copy['S_bias']
            
            pred_action = np.argmax([L_dot, R_dot, S_dot])
            pred_action = ['L', 'R', 'S'][pred_action]
                        
            # change configuration
            if pred_action == 'L':
                heads[stack[-1]] = buffer[0]
                stack.pop(-1)
            elif pred_action == 'R':
                heads[buffer[0]] = stack[-1]
                buffer[0] = stack.pop(-1)
            else:
                stack.append(buffer.pop(0))
        
        ## result
        res = sen
        for ix in range(len(res)):
            res[ix][6] = heads[ix]
        res.pop(0) # remove root
        results.append(res)
    
    return results


def write(results, filename):
    writeout = []
    for x in results:
        for line_ix in range(len(x)):
            x[line_ix][0] = str(x[line_ix][0])
            x[line_ix][6] = str(x[line_ix][6])
                
        for line_ix in range(len(x)):
            x[line_ix] = '\t'.join(x[line_ix])
        
        y = ''.join(x)
        writeout.append(y)

    writeout = '\n'.join(writeout) + '\n'
    with open(filename, 'w') as file:
        file.write(writeout)


def eval(ref_path,out_path):
        ref_file = open(ref_path, "r")
        refs = ref_file.readlines()
        tst_file = open(out_path, "r")
        tsts = tst_file.readlines()
        
        total = 0
        correct = 0
        for i, ref_line in enumerate(refs):
                tst_line = tsts[i]
                ref_line = ref_line.strip()
                tst_line = tst_line.strip()
                if len(ref_line) > 0:
                        (rnum, rname, rname1, rpos, rpos1, runder, rhead, rdep, rmore, rlast) = ref_line.split('\t')
                        (tnum, tname, tname1, tpos, tpos1, tunder, thead, tdep, tmore, tlast) = tst_line.split('\t')
                        total += 1
                        if rhead == thead:
                                correct += 1
                                
        print("%f%% (%d/%d)" % (float(correct)/total*100, correct, total))


# =============================================================================
# ## train
# features = {'L_bias': 0., 'R_bias': 0., 'S_bias': 0.}
# train('en.tr100', lr=1, n_iter=200)
# 
# ## predict
# write(predict('en.dev', features), 'result')
# 
# ## evaluation
# eval('en.dev', 'result')
# 
# ## remove file
# os.remove('result')
# =============================================================================

## simple grid search

if __name__ == "__main__":
        train1 = sys.argv[1]
        test   = sys.argv[2]
        testOutPut = sys.argv[3]
        features = {'L_bias': 0., 'R_bias': 0., 'S_bias': 0., 'D_bias':0.}
        train(train1, lr=1, n_iter=200)
        write(predict(test,features), testOutPut)
        # evaluation
        eval(test, testOutPut)
        # remove file
        os.remove(testOutPut)



    
    
    
    
    
    
    
    
    
    