
require('torch')
require('nn')

local ma = torch.class('MonkeyAgent')

function ma:actOnInput( input )
	t = torch.Tensor(1)
	t:random(1,3)
	return t
end

function ma:train( tuple )

end
