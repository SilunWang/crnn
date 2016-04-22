function lstm(h_t, h_d, prev_c, rnn_size)
	local all_input_sums = nn.CAddTable()({h_t, h_d})
	local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
	local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
	-- decode the gates
	local in_gate = nn.Sigmoid()(n1)
	local forget_gate = nn.Sigmoid()(n2)
	local out_gate = nn.Sigmoid()(n3)
	-- decode the write inputs
	local in_transform = nn.Tanh()(n4)
	-- perform the LSTM update
	local next_c           = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate,     in_transform})
	})
	-- gated cells form the output
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	return next_c, next_h
end



function makeGridLstmUnit(nIn, nHidden, dropout, n, should_tie_weights)
	--[[ Create LSTM unit, adapted from https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
	ARGS:
	- `nIn`      : integer, number of input dimensions
	- `nHidden`  : integer, number of hidden nodes
	- `dropout`  : boolean, if true apply dropout
	- `n`		   : integer, number of the depth
	RETURNS:
	- `lstmUnit` : constructed LSTM unit (nngraph module)
	]]
	dropout = dropout or 0

	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- input h for depth dimension
	table.insert(inputs, nn.Identity()()) -- input c for depth dimension
	for L = 1,n do
		table.insert(inputs, nn.Identity()()) -- prev_h[L] for time dimension
		table.insert(inputs, nn.Identity()()) -- prev_c[L] for time dimension
	end

	local shared_weights
	if should_tie_weights then shared_weights = {nn.Linear(nHidden, 4 * nHidden), nn.Linear(nHidden, 4 * nHidden)} end

	local outputs_t = {} -- Outputs being handed to the next time step along the time dimension
	local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension

	for L = 1,n do
		-- Take hidden and memory cell from previous time steps
		local prev_h_t = inputs[L*2+1]
		local prev_c_t = inputs[L*2+2]

		if L == 1 then
			-- We're in the first layer
			--prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) -- input_h_d: the starting depth dimension hidden state. We map a char into hidden space via a lookup table
			prev_h_d = inputs[1]
			prev_c_d = inputs[2] -- input_c_d: the starting depth dimension memory cell, just a zero vec.
		else 
			-- We're in the higher layers 2...N
			-- Take hidden and memory cell from layers below
			prev_h_d = outputs_d[((L-1)*2)-1]
			prev_c_d = outputs_d[((L-1)*2)]
			if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
		end

		-- Evaluate the input sums at once for efficiency
		local t2h_t = nn.Linear(nHidden, 4 * nHidden)(prev_h_t):annotate{name='i2h_'..L}
		local d2h_t = nn.Linear(nHidden, 4 * nHidden)(prev_h_d):annotate{name='h2h_'..L}

		-- Get transformed memory and hidden states pointing in the time direction first
		local next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, nHidden)

		-- Pass memory cell and hidden state to next timestep
		table.insert(outputs_t, next_h_t)
		table.insert(outputs_t, next_c_t)

		-- Evaluate the input sums at once for efficiency
		local t2h_d = nn.Linear(nHidden, 4 * nHidden)(next_h_t):annotate{name='i2h_'..L}
		local d2h_d = nn.Linear(nHidden, 4 * nHidden)(prev_h_d):annotate{name='h2h_'..L}

		-- See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
		-- The weights along the temporal dimension are already tied (cloned many times in train.lua)
		-- Here we can tie the weights along the depth dimension. Having invariance in computation
		-- along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
		-- See fig 4. to compare tied vs untied grid lstms on this task.
		if should_tie_weights then
			print("tying weights along the depth dimension")
			t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
			d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
		end

		-- Create the lstm gated update pointing in the depth direction.
		-- We 'prioritize' the depth dimension by using the updated temporal hidden state as input
		-- instead of the previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
		local next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, nHidden)

		-- Pass the depth dimension memory cell and hidden state to layer above
		table.insert(outputs_d, next_h_d)
		table.insert(outputs_d, next_c_d)
	end

	-- set up the decoder
	local top_h = outputs_d[#outputs_d - 1]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	--local proj = nn.Linear(nHidden, nOut)(top_h):annotate{name='decoder'}
	--local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
	--local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs_t, top_h)

	return nn.gModule(inputs, outputs_t)

end


local GridLstmLayer, parent = torch.class('nn.GridLstmLayer', 'nn.Module')


function GridLstmLayer:__init(nIn, nHidden, maxT, depth, dropout, reverse, tie_weight)
	--[[
	ARGS:
	- `nIn`     : integer, number of input dimensions
	- `nHidden` : integer, number of hidden nodes
	- `maxT`    : integer, maximum length of input sequence
	- `depth`   : The depth of 2d grid-lstm
	- `dropout` : boolean, if true apply dropout
	- `reverse` : boolean, if true the sequence is traversed from the end to the start
	]]
	parent.__init(self)

	self.dropout = dropout or 0
	self.reverse = reverse or false
	self.nHidden = nHidden
	self.maxT    = maxT
	self.depth = depth
	self.tie_weight = tie_weight or false

	self.output    = {}
	self.gradInput = {}

	-- LSTM unit and clones
	self.lstmUnit = makeGridLstmUnit(nIn, nHidden, self.dropout, self.depth, self.tie_weight)
	self.clones   = {}

	-- LSTM states
	self.initState = {} -- c_t, h_t
	for L = 1,2*self.depth do
		self.initState[L] = torch.CudaTensor()
	end

	self.initCellState = torch.CudaTensor()

	self:reset()
end


function GridLstmLayer:reset(stdv)
	local params, _ = self:parameters()
	for i = 1, #params do
		if i % 2 == 1 then -- weight
			params[i]:uniform(-0.08, 0.08)
		else -- bias
			params[i]:zero()
		end
	end
end


function GridLstmLayer:type(type)
	assert(#self.clones == 0, 'Function type() should not be called after cloning.')
	self.lstmUnit:type(type)
	return self
end


function GridLstmLayer:parameters()
	return self.lstmUnit:parameters()
end


function GridLstmLayer:training()
	self.train = true
	self.lstmUnit:training()
	for t = 1, #self.clones do self.clones[t]:training() end
end


function GridLstmLayer:evaluate()
	self.train = false
	self.lstmUnit:evaluate()
	for t = 1, #self.clones do self.clones[t]:evaluate() end
end

function subrange(t, first, last)
	local sub = {}
	for i=first,last do
		sub[#sub + 1] = t[i]
	end
	return sub
end


function GridLstmLayer:updateOutput(input)
	assert(type(input) == 'table')
	self.output = {}
	local T = #input
	local batchSize = input[1]:size(1)
	local n = self.depth

	for L = 1,2*n do
		self.initState[L]:resize(batchSize, self.nHidden):fill(0)
	end
	self.initCellState:resize(batchSize, self.nHidden):fill(0)

	if #self.clones == 0 then
		self.clones = cloneManyTimes(self.lstmUnit, self.maxT)
	end

	if not self.reverse then
		self.rnnState = {[0] = cloneList(self.initState, true)}
		self.initC = {}
		for t = 1, T do
			local lst
			self.initC[t] = self.initCellState:clone():zero()
			if self.train then
				lst = self.clones[t]:forward({input[t], self.initC[t], unpack(self.rnnState[t-1])})
			else
				lst = self.lstmUnit:forward({input[t], self.initC[t], unpack(self.rnnState[t-1])})
				lst = cloneList(lst)
			end
			self.rnnState[t] = subrange(lst, 1, 2*n)
			self.output[t] = lst[2*n+1]
		end
	else
		self.rnnState = {[T+1] = cloneList(self.initState, true)}
		self.initC = {}
		for t = T, 1, -1 do
			local lst
			self.initC[t] = self.initCellState:clone():zero()
			if self.train then
				lst = self.clones[t]:forward({input[t], self.initC[t], unpack(self.rnnState[t+1])})
			else
				lst = self.lstmUnit:forward({input[t], self.initC[t], unpack(self.rnnState[t+1])})
				lst = cloneList(lst)
			end
			self.rnnState[t] = subrange(lst, 1, 2*n);
			self.output[t] = lst[2*n+1]
		end
	end

	return self.output
end


function GridLstmLayer:updateGradInput(input, gradOutput)
	assert(#input == #gradOutput)
	local T = #input
	local n = self.depth

	if not self.reverse then
		self.drnnState = {[T] = cloneList(self.initState, true)} -- zero gradient for the last frame
		for t = T, 1, -1 do
			local doutput_t = gradOutput[t]
			table.insert(self.drnnState[t], doutput_t) -- dnext_c, dnext_h, doutput_t
			local dlst = self.clones[t]:updateGradInput({input[t], self.initC[t], unpack(self.rnnState[t-1])}, self.drnnState[t]) -- dx, dprev_c, dprev_h
			self.drnnState[t-1] = subrange(dlst, 3, 2*(n+1))
			self.gradInput[t] = dlst[1]
		end
	else
		self.drnnState = {[1] = cloneList(self.initState, true)}
		for t = 1, T do
			local doutput_t = gradOutput[t]
			table.insert(self.drnnState[t], doutput_t)
			local dlst = self.clones[t]:updateGradInput({input[t], self.initC[t], unpack(self.rnnState[t+1])}, self.drnnState[t])
			self.drnnState[t+1] = subrange(dlst, 3, 2*(n+1))
			self.gradInput[t] = dlst[1]
		end
	end

	return self.gradInput
end


function GridLstmLayer:accGradParameters(input, gradOutput, scale)
	local T = #input
	if not self.reverse then
		for t = 1, T do
			self.clones[t]:accGradParameters({input[t], self.initC[t], unpack(self.rnnState[t-1])}, self.drnnState[t], scale)
		end
	else
		for t = T, 1, -1 do
			self.clones[t]:accGradParameters({input[t], self.initC[t], unpack(self.rnnState[t+1])}, self.drnnState[t], scale)
		end
	end
end
