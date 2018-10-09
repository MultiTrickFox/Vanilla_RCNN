import torch
import random

hm_vectors = 4
vector_size = 13


max_prop_time = 20


default_filters = ((3,4),
                   (6,7,8),
                   (9,10,11))


#   Structure


def create_model(filters=default_filters):

    model = []
    hm_filters = len(filters)

    # layer : presence_1

    model.append(
        {
            'vr':torch.randn([hm_vectors,1], requires_grad=True),
            'ur':torch.randn([vector_size,vector_size], requires_grad=True),
            'br':torch.zeros([vector_size,1], requires_grad=True),

            'va':torch.randn([hm_vectors,1], requires_grad=True),
            'ua':torch.randn([vector_size,vector_size], requires_grad=True),
            'ba':torch.zeros([vector_size,1], requires_grad=True),

            'vs':torch.randn([hm_vectors,1], requires_grad=True),
            'bs':torch.zeros([vector_size,1], requires_grad=True),
        }
    )

    # layer : presence_2

    model.append(
        {
            'vr':torch.randn([hm_filters,1], requires_grad=True),
            'ur':torch.randn([vector_size,vector_size], requires_grad=True),
            'br':torch.zeros([vector_size,1], requires_grad=True),

            'va':torch.randn([hm_filters,1], requires_grad=True),
            'ua':torch.randn([vector_size,vector_size], requires_grad=True),
            'ba':torch.zeros([vector_size,1], requires_grad=True),

            'vs':torch.randn([hm_filters,1], requires_grad=True),
            'bs':torch.zeros([vector_size,1], requires_grad=True),
        }
    )

    # layer : convolution

    layer = {}

    for _,filter in enumerate(filters):
        layer['wf'+str(_)] = torch.randn([len(filter),1], requires_grad=True)

    model.append(layer)

    # layer : decision

    layer = {
        'vr':torch.randn([vector_size,1], requires_grad=True),
        'ur':torch.randn([vector_size,vector_size], requires_grad=True),
        'br':torch.zeros([vector_size,1], requires_grad=True),

        'vf':torch.randn([vector_size,1], requires_grad=True),
        'uf':torch.randn([vector_size,vector_size], requires_grad=True),
        'bf':torch.zeros([vector_size,1], requires_grad=True),

        'va':torch.randn([vector_size,1], requires_grad=True),
        'ua':torch.randn([vector_size,vector_size], requires_grad=True),
        'ba':torch.zeros([vector_size,1], requires_grad=True),

        'vs':torch.randn([vector_size,1], requires_grad=True),
        'us':torch.randn([vector_size,vector_size], requires_grad=True),
        'bs':torch.zeros([vector_size,1], requires_grad=True),
    }

    for _ in range(1,hm_vectors):

        str_ = '_'+str(_)

        layer['vr2'+str_] = torch.randn([vector_size,1], requires_grad=True)
        layer['va2'+str_] = torch.randn([vector_size,1], requires_grad=True)
        layer['vf2'+str_] = torch.randn([vector_size,1], requires_grad=True)
        layer['vs2'+str_] = torch.randn([vector_size,1], requires_grad=True)

        layer['ur2'+str_] = torch.randn([vector_size,vector_size], requires_grad=True)
        layer['ua2'+str_] = torch.randn([vector_size,vector_size], requires_grad=True)
        layer['uf2'+str_] = torch.randn([vector_size,vector_size], requires_grad=True)
        layer['us2'+str_] = torch.randn([vector_size,vector_size], requires_grad=True)

    model.append(layer)

    return model


# Forward Prop


def forward_prop(model, sequence, context=None, gen_seed=None, gen_iterations=None, filters=default_filters, dropout=0.0):


    #   listen


    vocab_seq, oct_seq, dur_seq, vol_seq = sequence
    states = [context] if context is not None else init_states(model)
    outputs = []

    for t in range(len(sequence[0])):

        output, state = prop_func(model, [vocab_seq[t], oct_seq[t], dur_seq[t], vol_seq[t]], states[-1], filters=filters, dropout=dropout)

        outputs.append(output)
        states.append(state)


    #   generate


    states = [states[-1]]
    outputs = [gen_seed] if gen_seed is not None else [outputs[-1]]     # sequence, an array of outputs, where out[0] = vocab, out[1] = rhythm

    if gen_iterations is None:

        t = 0
        while t < max_prop_time and not stop_cond(outputs[-1]):

            output, state = prop_func(model, outputs[-1], states[-1], filters=filters, dropout=0.0)

            outputs.append(output)
            states.append(state)
            t += 1

    else:

        for t in range(gen_iterations):

            output, state = prop_func(model, outputs[-1], states[-1], filters=filters, dropout=dropout)

            states.append(output)
            outputs.append(state)


    del outputs[0]
    return outputs


def prop_func(model, sequence_t, context_t, filters, dropout):

    produced_outputs = []
    produced_context = []

    # layer : presence_1

    input = torch.stack(sequence_t, 1)

    remember = torch.sigmoid(
        torch.matmul(input, model[0]['vr']) +
        torch.matmul(model[0]['ur'], context_t[0]) +
        model[0]['br']
    )

    attention = torch.tanh(
        torch.matmul(input, model[0]['va']) +
        torch.matmul(model[0]['ua'], context_t[0]) +
        model[0]['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(input, model[0]['vs']) +
        attention * context_t[0] +
        model[0]['bs']
    )

    state = remember * short_mem + (1-remember) * context_t[0]

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = torch.Tensor(0)

    produced_context.append(state)

    # layer : convolution

    input = produced_context[-1]

    convolutions = []

    for _,filter in enumerate(filters):
        convolution = vector_convolve(input, filter_values=filter, filter_weights=model[2]['wf'+str(_)])
        convolutions.append(torch.Tensor(convolution))

    # layer : presence_2

    input = torch.stack(convolutions, 1)

    remember = torch.sigmoid(
        torch.matmul(input, model[1]['vr']) +
        torch.matmul(model[1]['ur'], context_t[1]) +
        model[1]['br']
    )

    attention = torch.tanh(
        torch.matmul(input, model[1]['va']) +
        torch.matmul(model[1]['ua'], context_t[1]) +
        model[1]['ba']
    )

    short_mem = torch.tanh(
        torch.matmul(input, model[1]['vs']) +
        attention * context_t[1] +
        model[1]['bs']
    )

    state = remember * short_mem + (1-remember) * context_t[1]

    if dropout != 0.0:
        drop = random.choices(range(len(state)), k=int(len(state) * dropout))
        for _ in drop: state[_] = torch.Tensor(0)

    produced_context.append(state)

    # layer : decision

    input1 = produced_context[-1]
    state1 = context_t[0]

    layer_produced_context = []

    for _ in range(1, hm_vectors):

        str_ = '_'+str(_)
        print(len(context_t[-1]))
        input2 = sequence_t[_]
        state2 = context_t[-1][_]

        remember = torch.sigmoid(
            input1 * model[-1]['vr'] +
            input2 * model[-1]['vr2'+str_] +
            torch.matmul(model[-1]['ur'], state1) +
            torch.matmul(model[-1]['ur2'+str_], state2)
        )

        forget = torch.sigmoid(
            input1 * model[-1]['vf'] +
            input2 * model[-1]['vf2'+str_] +
            torch.matmul(model[-1]['uf'], state1) +
            torch.matmul(model[-1]['uf2'+str_], state2)
        )

        attention = torch.tanh(
            input1 * model[-1]['vr'] +
            input2 * model[-1]['vr2'+str_] +
            torch.matmul(model[-1]['ur'], state1) +
            torch.matmul(model[-1]['ur2'+str_], state2)
        )

        short_mem = torch.tanh(
            input1 * model[-1]['vs'] +
            input2 * model[-1]['vs2'+str_] +
            torch.matmul(model[-1]['us'], state1) +
            torch.matmul(model[-1]['us2'+str_], state2)
        )

        layer_produced_context.append(remember * short_mem + forget * context_t[-1][_])
        produced_outputs.append(attention * torch.tanh(layer_produced_context[-1]))
    produced_context.append(layer_produced_context)

    return produced_outputs, produced_context


#   Math Operations


def vector_convolve(vector, filter_values, filter_weights):

    to_convolve = []
    convolution = []

    for _ in range(vector_size):
        to_convolve.append(torch.Tensor([vector[(_+i) % vector_size] for i in filter_values]))

    for _,element in enumerate(to_convolve):
        convolution.append((element * filter_weights).sum())

    return convolution


def loss_wrt_distance(output_seq, label_seq):
    sequence_losses = []

    for t in range(len(label_seq)):
        lbl = label_seq[t]
        pred = output_seq[t]

        for _,lbl_e in enumerate(lbl):
            pred_e = pred[_]

            loss = lbl_e - pred_e
            sequence_losses.append(loss.sum())

    return sequence_losses


#   Optimization


def update_gradients(loss_nodes):
    for node in loss_nodes:
        node.backward(retain_graph=True)


def update_model(model, batch_size=1, learning_rate=0.001):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight += learning_rate * weight.grad / batch_size
                    weight.grad = None


def update_model_momentum(model, moments, batch_size=1, alpha=0.9, beta=0.1):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    moments[_][__] = alpha * moments[_][__] + beta * weight.grad
                    weight += moments[_][__]
                    weight.grad = None


def update_model_rmsprop(model, accu_grads, batch_size=1, lr=0.01, alpha=0.9):
    with torch.no_grad():
        for _,layer in enumerate(model):
            for __,weight in enumerate(layer.values()):
                if weight.grad is not None:
                    weight.grad /= batch_size
                    accu_grads[_][__] = alpha * accu_grads[_][__] + (1 - alpha) * weight.grad**2
                    weight += lr * weight.grad / (torch.sqrt(accu_grads[_][__]) + 1e-8)
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

                    weight += lr * moment_hat / (torch.sqrt(sum(accugrad_hat)) + 1e-8)
                    weight.grad = None


#   Helpers


def init_states(model):
    states_t0 = [torch.randn([vector_size,1], requires_grad=True) for _ in range(len(model)-2)]
    states_t0.append([torch.randn([vector_size,1], requires_grad=True) for _ in range(hm_vectors)])

    return [states_t0]


import res ; stop_dur = res.SPLIT_DURATION

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
