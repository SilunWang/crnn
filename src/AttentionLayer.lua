require 'mixture'
require 'mask_table'

function makeAttUnit(nIn, nHidden, dropout)
    --[[ Create Attention LSTM unit, adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
    ARGS:
      - `nIn`      : integer, number of input dimensions
      - `nHidden`  : integer, number of hidden nodes
      - `dropout`  : boolean, if true apply dropout
    RETURNS:
      - `AttUnit` : constructed Attention LSTM unit (nngraph module)
    ]]
    dropout = dropout or 0

    -- x:       a_i   batch * 1 * nHidden
    -- a:       {a_i} batch * seq_len * nHidden
    -- prev_c:  c_t-1 batch * 1 * nHidden
    -- prev_h:  h_t-1 batch * 1 * nHidden
    local x, a, prev_c, prev_h = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    local inputs = {x, a, prev_c, prev_h}

    -- Construct the unit structure
    -- apply dropout, if any
    if dropout > 0 then 
        x = nn.Dropout(dropout)(x) 
        a = nn.Dropout(dropout)(a)
    end

    -- evaluate the input sums at once for efficiency
    local i2h            = nn.Linear(nIn,     4*nHidden)(x)
    local h2h            = nn.Linear(nHidden, 4*nHidden)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk  = nn.Narrow(2, 1, 3*nHidden)(all_input_sums)
    sigmoid_chunk        = nn.Sigmoid()(sigmoid_chunk)
    local in_gate        = nn.Narrow(2,           1, nHidden)(sigmoid_chunk)
    local forget_gate    = nn.Narrow(2,   nHidden+1, nHidden)(sigmoid_chunk)
    local out_gate       = nn.Narrow(2, 2*nHidden+1, nHidden)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform   = nn.Narrow(2, 3*nHidden+1, nHidden)(all_input_sums)
    in_transform         = nn.Tanh()(in_transform)

    -- e = f({a_i}, h_t-1): batch * seq_len
    local dot            = nn.Mixture(3){prev_h, a}
    -- weight = exp(e)/sum {exp(e)}: batch * seq_len
    local weight         = nn.SoftMax()(dot)  

    -- perform the LSTM update
    local next_c         = nn.CAddTable()({
                               nn.CMulTable()({forget_gate, prev_c}),
                               nn.CMulTable()({in_gate    , in_transform})
                               })
    -- gated cells from the output
    local next_h         = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    -- a_t: batch * nHidden * seq_len
    local a_t            = nn.Transpose({2, 3})(a)
    -- z_t = sum(weight[i] * a[i])  batch * nHidden
    local z_t            = nn.Mixture(3)({weight, a_t})
    -- wz = W * z_t
    -- local wz             = nn.Linear(nHidden, nHidden)(z_t)
    -- wh = W * h_t
    -- local wh             = nn.Linear(nHidden, nHidden)(next_h)
    -- y = W1 * z_t + W2 * h_t
    -- local y              = nn.Tanh()(nn.CAddTable()({wz, wh}))
    local y = nn.Tanh()(nn.Linear(nHidden * 2, nHidden)(nn.JoinTable(2)({z_t, next_h})))

    -- there will be 3 outputs
    local outputs = {next_c, next_h, y}

    local AttUnit = nn.gModule(inputs, outputs)
    return AttUnit
end


local AttentionLayer, parent = torch.class('nn.AttentionLayer', 'nn.Module')


function AttentionLayer:__init(nIn, nHidden, maxT, dropout, reverse)
    --[[
    ARGS:
      - `nIn`     : integer, number of input dimensions
      - `nHidden` : integer, number of hidden nodes
      - `maxT`    : integer, maximum length of input sequence
      - `dropout` : boolean, if true apply dropout
      - `reverse` : boolean, if true the sequence is traversed from the end to the start
    ]]
    parent.__init(self)

    self.dropout = dropout or 0
    self.reverse = reverse or false
    self.nHidden = nHidden
    self.maxT    = maxT

    self.output    = {}
    self.gradInput = {}

    -- LSTM unit and clones
    self.AttUnit = makeAttUnit(nIn, nHidden, self.dropout)
    self.clones   = {}

    -- LSTM states
    self.initState = {torch.CudaTensor(), torch.CudaTensor()} -- c, h

    self:reset()
end


function AttentionLayer:reset(stdv)
    local params, _ = self:parameters()
    for i = 1, #params do
        if i % 2 == 1 then -- weight
            params[i]:uniform(-0.08, 0.08)
        else -- bias
            params[i]:zero()
        end
    end
end


function AttentionLayer:type(type)
    assert(#self.clones == 0, 'Function type() should not be called after cloning.')
    self.AttUnit:type(type)
    return self
end


function AttentionLayer:parameters()
    return self.AttUnit:parameters()
end


function AttentionLayer:training()
    self.train = true
    self.AttUnit:training()
    for t = 1, #self.clones do self.clones[t]:training() end
end


function AttentionLayer:evaluate()
    self.train = false
    self.AttUnit:evaluate()
    for t = 1, #self.clones do self.clones[t]:evaluate() end
end


function AttentionLayer:updateOutput(input)
    self.output = {}
    local T = input:size(2)
    local batchSize = input:size(1)
    self.initState[1]:resize(batchSize, self.nHidden):fill(0)
    self.initState[2]:resize(batchSize, self.nHidden):fill(0)
    if #self.clones == 0 then
        self.clones = cloneManyTimes(self.AttUnit, self.maxT)
    end

    if not self.reverse then
        self.rnnState = {[0] = cloneList(self.initState, true)}
        for t = 1, T do
            local lst
            if self.train then
                lst = self.clones[t]:forward({input:select(2,t), input, unpack(self.rnnState[t-1])})
            else
                lst = self.AttUnit:forward({input:select(2,t), input, unpack(self.rnnState[t-1])})
                lst = cloneList(lst)
            end
            self.rnnState[t] = {lst[1], lst[2]} -- next_c, next_h
            self.output[t] = lst[3]
        end
    else
        self.rnnState = {[T+1] = cloneList(self.initState, true)}
        for t = T, 1, -1 do
            local lst
            if self.train then
                lst = self.clones[t]:forward({input:select(2,t), input, unpack(self.rnnState[t+1])})
            else
                lst = self.AttUnit:forward({input:select(2,t), input, unpack(self.rnnState[t+1])})
                lst = cloneList(lst)
            end
            self.rnnState[t] = {lst[1], lst[2]}
            self.output[t] = lst[3]
        end
    end
    return self.output
end


function AttentionLayer:updateGradInput(input, gradOutput)
    assert(input:size(2) == #gradOutput)
    local T = input:size(2)
    self.gradInput = torch.CudaTensor(input:size())
    if not self.reverse then
        self.drnnState = {[T] = cloneList(self.initState, true)} -- zero gradient for the last frame
        for t = T, 1, -1 do
            local doutput_t = gradOutput[t]
            table.insert(self.drnnState[t], doutput_t) -- dnext_c, dnext_h, doutput_t
            local dlst = self.clones[t]:updateGradInput({input:select(2,t), input, unpack(self.rnnState[t-1])}, self.drnnState[t]) -- dx, da, dprev_c, dprev_h
            self.drnnState[t-1] = {dlst[3], dlst[4]}
            self.gradInput:select(2,t):copy(dlst[1])
        end
    else
        self.drnnState = {[1] = cloneList(self.initState, true)}
        for t = 1, T do
            local doutput_t = gradOutput[t]
            table.insert(self.drnnState[t], doutput_t)
            local dlst = self.clones[t]:updateGradInput({input:select(2,t), input, unpack(self.rnnState[t+1])}, self.drnnState[t])
            self.drnnState[t+1] = {dlst[3], dlst[4]}
            self.gradInput:select(2,t):copy(dlst[1])
        end
    end
    return self.gradInput
end


function AttentionLayer:accGradParameters(input, gradOutput, scale)
    local T = input:size(2)
    if not self.reverse then
        for t = 1, T do
            self.clones[t]:accGradParameters({input:select(2,t), input, unpack(self.rnnState[t-1])}, self.drnnState[t], scale)
        end
    else
        for t = T, 1, -1 do
            self.clones[t]:accGradParameters({input:select(2,t), input, unpack(self.rnnState[t+1])}, self.drnnState[t], scale)
        end
    end
end
