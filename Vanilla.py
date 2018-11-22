import torch
import random

hm_vectors = 4
vector_size = 13


max_prop_time = 20


default_layers = (8, 5, 10)

default_filters = (

    (3, 4),         # 3rd range
    (6, 7, 8),      # 5th range
    (9, 10, 11),    # 6-7th range
    (6, 7, 1, 2),   # 7-2th range

    # (4, 7),         # major chord   @todo: remove this part if too much loss.
    # (3, 7),         # minor chord
    # (9, 4),         # alt-minor chord

                   )

default_loss_multipliers = (1, 1, 1, 1)


#   Structure


def create_model(filters=default_filters,layers=default_layers):

    model = []
    hm_filters = len(filters)

    # layer : presence_near

    model.append(
        {
            'vr': torch.randn([1,hm_vectors], requires_grad=True),
            'ur': torch.randn([vector_size,vector_size], requires_grad=True),
            'br': torch.zeros([1,vector_size], requires_grad=True),

            'vf': torch.randn([1,hm_vectors], requires_grad=True),
            'uf': torch.randn([1,vector_size], requires_grad=True),
            'bf': torch.zeros([1,vector_size], requires_grad=True),

            'va': torch.randn([1,hm_vectors], requires_grad=True),
            'ua': torch.randn([1,vector_size], requires_grad=True),
            'ba': torch.zeros([1,vector_size], requires_grad=True),

            'vs': torch.randn([1,hm_vectors], requires_grad=True),
            'us': torch.randn([1,vector_size], requires_grad=True),
            'bs': torch.zeros([1,vector_size], requires_grad=True),
        }
    )

    # layer : convolutions

    layer = {}
    for _,filter in enumerate(filters):
        layer['wf'+str(_)] = torch.randn([len(filter),1], requires_grad=True)

    model.append(layer)

    # layer : presence_far

    in_size = vector_size*hm_filters
    # mid_size = vector_size*2 # int(in_size*3/4)

    model.append(
        {
            'vr': torch.randn([1,hm_filters], requires_grad=True),
            'vr2': torch.randn([in_size,vector_size], requires_grad=True),
            # 'vr2.2': torch.randn([mid_size,vector_size], requires_grad=True),
            'ur': torch.randn([1,vector_size], requires_grad=True),
            'br': torch.zeros([1,vector_size], requires_grad=True),

            'va': torch.randn([1,hm_filters], requires_grad=True),
            'va2': torch.randn([in_size,vector_size], requires_grad=True),
            # 'va2.2': torch.randn([mid_size,vector_size], requires_grad=True),
            'ua': torch.randn([1,vector_size], requires_grad=True),
            'ba': torch.zeros([1,vector_size], requires_grad=True),

            'vs': torch.randn([1,hm_filters], requires_grad=True),
            'vs2': torch.randn([in_size,vector_size], requires_grad=True),
            # 'vs2.2': torch.randn([mid_size,vector_size], requires_grad=True),
            'bs': torch.zeros([1,vector_size], requires_grad=True),
        }
    )

    # layer : decision

    hm_layers = len(layers)
    if layers[-1] != vector_size:
        layers = layers[:-1] + (vector_size, )

    for _,layer_size in enumerate(layers):

        layer = {}

        if _ == 0:
            in_size = vector_size
            out_size = layer_size
            
        elif _ == hm_layers-1:
            in_size = layers[_-1]
            out_size = vector_size

        else:
            in_size = layers[_-1]
            out_size = layer_size

            # initial layer
        if _ == 0:

            for __ in range(1,hm_vectors):
                str_ = '_'+str(__)

                layer['vr' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['ur' + str_] = torch.randn([layer_size, layer_size], requires_grad=True)
                layer['wr' + str_] = torch.randn([layer_size], requires_grad=True)

                layer['va' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['ua' + str_] = torch.randn([layer_size, layer_size], requires_grad=True)
                layer['wa' + str_] = torch.randn([layer_size], requires_grad=True)

                layer['vs' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['ws' + str_] = torch.randn([layer_size], requires_grad=True)

            # final layer
        elif _ == hm_layers-1:

            in_size2 = layers[_ - 2]

            for __ in range(1,hm_vectors):
                str_ = '_'+str(__)

                layer['vr' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['ur' + str_] = torch.randn([1, layer_size], requires_grad=True)
                layer['wif_r' + str_] = torch.randn([in_size2, layer_size], requires_grad=True)
                layer['wr' + str_] = torch.randn([layer_size], requires_grad=True)

                layer['va' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['ua' + str_] = torch.randn([1, layer_size], requires_grad=True)
                layer['ua2' + str_] = torch.randn([layer_size, layer_size], requires_grad=True)
                layer['wif_a' + str_] = torch.randn([in_size2, layer_size], requires_grad=True)
                layer['wa' + str_] = torch.randn([layer_size], requires_grad=True)

                layer['vs' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['us' + str_] = torch.randn([1, layer_size], requires_grad=True)
                layer['ws' + str_] = torch.randn([layer_size, layer_size], requires_grad=True)

                layer['vf' + str_] = torch.randn([in_size, layer_size], requires_grad=True)
                layer['uf' + str_] = torch.randn([1, layer_size], requires_grad=True)
                layer['wif_f' + str_] = torch.randn([in_size2, layer_size], requires_grad=True)
                layer['wf' + str_] = torch.randn([layer_size], requires_grad=True)

                layer['wo' + str_] = torch.randn([layer_size, out_size], requires_grad=True)
                layer['bo' + str_] = torch.zeros([out_size], requires_grad=True)

            # middle layers
        else:

            layer['vr'] = torch.randn([in_size, layer_size], requires_grad=True)
            layer['vr2'] = torch.randn([vector_size, layer_size], requires_grad=True)
            layer['ur'] = torch.randn([layer_size], requires_grad=True)

            layer['wr'] = torch.randn([layer_size], requires_grad=True)
            layer['br'] = torch.zeros([layer_size], requires_grad=True)


            layer['va'] = torch.randn([in_size, layer_size], requires_grad=True)
            layer['va2'] = torch.randn([vector_size, layer_size], requires_grad=True)
            layer['ua'] = torch.randn([layer_size], requires_grad=True)

            layer['wa'] = torch.randn([layer_size], requires_grad=True)
            layer['ba'] = torch.zeros([layer_size], requires_grad=True)


            layer['vs'] = torch.randn([in_size, layer_size], requires_grad=True)
            layer['vs2'] = torch.randn([vector_size, layer_size], requires_grad=True)

            layer['ws'] = torch.randn([layer_size], requires_grad=True)
            layer['bs'] = torch.zeros([layer_size], requires_grad=True)


        layer['br'] = torch.zeros([1, out_size], requires_grad=True)
        layer['bf'] = torch.zeros([1, out_size], requires_grad=True)
        layer['ba'] = torch.zeros([1, out_size], requires_grad=True)
        layer['bs'] = torch.zeros([1, out_size], requires_grad=True)


        model.append(layer)


    return model



# Forward Prop


def forward_prop(model, sequence, actual_output=None, context=None, gen_seed=None, gen_iterations=None, filters=default_filters, dropout=0.0):


    #   listen


    states = [context] if context is not None else init_states(model)
    outputs = []

    for t in range(len(sequence[0])):

        if actual_output is not None:
            output, state = forward_prop_t(model, sequence[t], states[-1], filters=filters, dropout=dropout, output_t=actual_output[t])
        else:
            output, state = forward_prop_t(model, sequence[t], states[-1], filters=filters, dropout=dropout)

        outputs.append(output)
        states.append(state)


    #   generate


    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]     # sequence, an array of outputs, where out[0] = vocab, out[1] = rhythm

    if gen_iterations is None:

        t = 0
        while t < max_prop_time and not stop_cond(outputs[-1]):

            output, state = forward_prop_t(model, outputs[-1], states[-1], filters=filters, dropout=0.0)

            outputs.append(output)
            states.append(state)
            t += 1

    else:

        for t in range(gen_iterations):

            output, state = forward_prop_t(model, outputs[-1], states[-1], filters=filters, dropout=dropout)

            outputs.append(output)
            states.append(state)


    del outputs[0]
    return outputs


def forward_prop_t(model, sequence_t, context_t, filters, dropout, output_t=None):

    produced_outputs = []
    produced_context = []

    # layer : presence_near

    layer = model[0]
    state = context_t[0]
    input = torch.stack(sequence_t, 0)

    remember = torch.sigmoid(
        torch.matmul(layer['vr'], input) +
        torch.matmul(state, layer['ur']) +
        layer['br']
    )

    # forget = torch.sigmoid(
    #    torch.matmul(layer['vf'], input) +
    #     torch.mul(state, layer['uf']) +
    #     layer['bf']
    # )

    attention = torch.tanh(
        torch.matmul(layer['va'], input) +
        torch.mul(state, layer['ua']) +
        layer['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(layer['vs'], input) +
        # torch.matmul(state, layer['us']) +
        attention * state +
        layer['bs']
    )

    # state = remember * short_mem + forget * state
    state = remember * short_mem + (1 - remember) * state
    # state_focused = (attention * torch.tanh(state))

    # layer_produced_context.append()
    # produced_outputs.append((attention * torch.tanh(layer_produced_context[-1])).squeeze(dim=0))

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = 0

    produced_context.append(state)

    # layer : convolution

    layer = model[1]
    input = produced_context[0]

    convolutions = [vector_convolve(input, filter_values=filter, filter_weights=layer['wf'+str(_)])
                    for _, filter in enumerate(filters)]

    # layer : presence_far

    layer = model[2]
    state = context_t[1]
    # input = torch.stack(convolutions, 0)
    input2 = torch.cat(convolutions).unsqueeze(0)

    remember = torch.sigmoid(
        # torch.matmul(layer['vr'], input) +
        torch.mul(state, layer['ur']) +
        # torch.matmul(torch.tanh(torch.matmul(input2, layer['vr2'])), layer['vr2.2']) +
        torch.matmul(input2, layer['vr2']) +
        layer['br']
    )

    attention = torch.tanh(
        # torch.matmul(layer['va'], input) +
        torch.mul(state, layer['ua']) +
        # torch.matmul(torch.tanh(torch.matmul(input2, layer['va2'])), layer['va2.2']) +
        torch.matmul(input2, layer['va2']) +
        layer['ba']
    )

    short_mem = torch.tanh(
        # torch.matmul(layer['vs'], input) +
        attention * state +
        # torch.matmul(torch.tanh(torch.matmul(input2, layer['vs2'])), layer['vs2.2']) +
        torch.matmul(input2, layer['vs2']) +
        layer['bs']
    )

    state = remember * short_mem + (1-remember) * state

    produced_outputs.append(state.squeeze(dim=0))

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = 0

    produced_context.append(state)

    # layer : decision

    decision_outputs = []

    hm_layers = len(model)-3

    for _ in range(hm_layers):

        layer = model[_+3]
        states = context_t[_+2]

        decision_outputs.append([])
        produced_context.append([])

        # initial layer
        if _ == 0:

            for __ in range(1,hm_vectors):

                str_ = '_'+str(__)

                input = sequence_t[__]
                state = states[__-1]

                remember = torch.sigmoid(
                    layer['wr' + str_] *
                    (torch.matmul(input, layer['vr' + str_]) +
                     torch.matmul(state, layer['ur' + str_])
                     )
                    # + layer['br']
                )

                attention = torch.sigmoid(
                    layer['wa' + str_] *
                    (torch.matmul(input, layer['va' + str_]) +
                     torch.matmul(state, layer['ua' + str_])
                     )
                    # + layer['ba']
                )

                short_mem = torch.tanh(
                    layer['ws' + str_] *
                    (torch.matmul(input, layer['vs' + str_]) +
                     attention * state
                     )
                    # + layer['bs']
                )

                state = remember * short_mem + (1 - remember) * state   # forget * state
                output = state                                          # (attention * torch.tanh(state))

                if dropout != 0.0:
                    drop = random.choices(range(len(output)), k=int(len(output) * dropout))
                    for _ in drop: output[_] = 0

                produced_context[-1].append(state)
                decision_outputs[-1].append(output)


        # final layer - lstm
        elif _ == hm_layers-1:

            input = decision_outputs[-2][0]

            for __ in range(1,hm_vectors):

                input2 = decision_outputs[-3][__-1]
                state = states[__-1]

                str_ = '_'+str(__)

                remember = torch.sigmoid(
                    layer['wr' + str_] *
                    (torch.matmul(input, layer['vr' + str_]) +
                     state * layer['ur' + str_] +
                     torch.matmul(input2, layer['wif_r' + str_])
                     )
                    # + layer['br']
                )

                forget = torch.sigmoid(
                    layer['wf' + str_] *
                    (torch.matmul(input, layer['vf' + str_]) +
                     state * layer['uf' + str_] +
                     torch.matmul(input2, layer['wif_f' + str_])
                     )
                    # + layer['bf']
                )

                attention = torch.sigmoid(
                    layer['wa' + str_] *
                    (torch.matmul(input, layer['va' + str_]) +
                     state * layer['ua' + str_] +
                     # torch.matmul(layer['ua2' + str_], out_context_t[_]) +
                     torch.matmul(input2, layer['wif_a' + str_])
                     )
                    # + layer['ba']
                )

                short_mem = torch.tanh(
                    torch.matmul(
                                 (torch.matmul(input, layer['vs' + str_]) +
                                  state * layer['us' + str_]
                                  ),
                        layer['ws' + str_]
                                 )
                    # + layer['bs']
                )

                state = remember * short_mem + forget * state
                produced_context[-1].append(state)
                output = (attention * torch.tanh(state))
                decision_outputs[-1].append(output)
                out = torch.sigmoid(torch.matmul(output, layer['wo'+str_]) + layer['bo'+str_])
                produced_outputs.append(out.squeeze(0))


        # middle layers - gru
        else:

            input = sum(produced_context[-2])
            # input = decision_outputs[-1]
            input2 = output_t[0] if output_t is not None else produced_outputs[0].unsqueeze(0)
            # input2 = produced_outputs[0].unsqueeze(0)
            # input2 = torch.tensor(produced_outputs[0].unsqueeze(0), requires_grad=False)

            state = states[0]

            remember = torch.sigmoid(
                layer['wr'] *
                (torch.matmul(input, layer['vr']) +
                 torch.matmul(input2, layer['vr2']) +
                 layer['ur'] * state
                 )
                # + layer['br']
            )

            attention = torch.sigmoid(
                layer['wa'] *
                (torch.matmul(input, layer['va']) +
                 torch.matmul(input2, layer['va2']) +
                 layer['ua'] * state
                 )
                # + layer['ba']
            )

            short_mem = torch.tanh(
                layer['ws'] *
                (torch.matmul(input, layer['vs']) +
                 torch.matmul(input2, layer['vs2']) +
                 attention * state
                 )
                + layer['bs']
            )

            state = remember * short_mem + (1-remember) * state
            output = state

            if dropout != 0.0:
                drop = random.choices(range(len(output)), k=int(len(output) * dropout))
                for _ in drop: output[_] = 0

            produced_context[-1].append(state)
            decision_outputs[-1].append(output)


    return produced_outputs, produced_context


#   Math Operations


def vector_convolve(vector, filter_values, filter_weights):

    to_convolve = []
    convolution = []

    for _ in range(vector_size):
        conv_vect = [vector[:,(_+i) % vector_size] for i in filter_values]
        to_convolve.append(torch.cat(conv_vect))

    for element in to_convolve:
        convolution.append((element * filter_weights).sum())

    return torch.Tensor(convolution)


def loss_wrt_distance(output_seq, label_seq):

    sequence_losses = [[],[],[],[]]

    for t in range(len(label_seq)):
        lbl = label_seq[t]
        pred = output_seq[t]

        for _,lbl_e in enumerate(lbl):
            pred_e = pred[_]

            loss = (lbl_e - pred_e).pow(2)
            # loss = lbl_e * -(torch.log(pred_e)) if lbl_e != 0 else 0

            sequence_losses[_].append(loss.sum())

    return sequence_losses


#   Optimization


def update_gradients(sequence_loss, loss_multipliers=default_loss_multipliers):
    for _,node in enumerate(sequence_loss):
        multiplier = torch.Tensor([loss_multipliers[_]])
        for time_step in node:
           time_step.backward(retain_graph=True,
                              gradient=multiplier
                              )


def update_model(model, batch_size=1, learning_rate=0.001):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight -= learning_rate * weight.grad / batch_size
                    weight.grad = None


def update_model_momentum(model, moments, batch_size=1, alpha=0.9, beta=0.1):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    moments[_][__] = alpha * moments[_][__] + beta * weight.grad
                    weight -= moments[_][__]
                    weight.grad = None


def update_model_rmsprop(model, accu_grads, batch_size=1, lr=0.01, alpha=0.9):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    accu_grads[_][__] = alpha * accu_grads[_][__] + (1 - alpha) * weight.grad**2
                    weight -= lr * weight.grad / (torch.sqrt(accu_grads[_][__]) + 1e-8)
                    weight.grad = None


def update_model_adam(model, accugrads, moments, epoch_nr, batch_size=1, lr=0.001, alpha_moments=0.9, alpha_accugrads=0.999):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size

                    moments[_][__] = alpha_moments * moments[_][__] + (1 - alpha_moments) * weight.grad
                    accugrads[_][__] = alpha_accugrads * accugrads[_][__] + (1 - alpha_accugrads) * weight.grad ** 2

                    moment_hat = moments[_][__] / (1 - alpha_moments ** epoch_nr)
                    accugrad_hat = accugrads[_][__] / (1 - alpha_accugrads ** epoch_nr)

                    weight -= lr * moment_hat / (torch.sqrt(sum(accugrad_hat)) + 1e-8)
                    weight.grad = None


#   Helpers


def init_states(model):

    states_t0 = [torch.zeros([1,vector_size], requires_grad=True) for _ in range(2)]    # presence _near & _far

    # [states_t0.append(torch.randn([1,layer['vr'].size()[1]], requires_grad=True))       # decision deep-layer(s)
    #     for layer in model[3:]]

    hm_layers = len(model)-3

    for _,layer in enumerate(model[3:]):

        if _ == 0:

            states_t0.append([torch.zeros([1,layer['vr_1'].size()[1]], requires_grad=False) for __ in range(1,hm_vectors)])

        elif _ == hm_layers-1:

            states_t0.append([torch.zeros([1,layer['vr_1'].size()[1]], requires_grad=False) for __ in range(1,hm_vectors)])

        else:

            states_t0.append([torch.zeros([1,layer['vr'].size()[1]], requires_grad=False) for __ in range(1,hm_vectors)])


    return [states_t0]


stop_dur = 2.0


def stop_cond(output_t):

    durations = output_t[2]

    for dur in durations:
        if float(dur) >= stop_dur:
            return True
    return False


def return_grads(model):
    grads = []
    for _,layer in enumerate(model):
        for w in layer:
            grads.append(layer[w].grad)
    return grads


def apply_grads(model, grads):
    ctr = 0
    for _,layer in enumerate(model):
        for w in layer:
            this_grad = grads[ctr]
            if layer[w].grad is None: layer[w].grad = this_grad
            else: layer[w].grad += this_grad

            ctr +=1
