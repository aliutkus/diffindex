import torch
from torch import nn
import asteroid.filterbanks as fb
from warnings import warn
from torchinterp1d import Interp1d
from torchsearchsorted import searchsorted


class Indexer(nn.Module):
    def __init__(self,
                 items,
                 cyclic=False,
                 dtype=torch.float32,
                 stiffness=2.):
        """
        Differentiable indexer

        parmeters:
        items: a list of torch modules or a pytorch Tensor.
            in the latter case, the items will be taken along the
            first dimension.
        mode: str, 'cyclic' or 'clamped'
            describes how to handle indices beyond [0, 1].
        stiffness: float
            in case the indexer is called with indices that
            require grad, soft-indexing is used to make it differentiable.
            This is done by approximating the stair-like indexing by a
            group of logistic functions.
            The higher stiffness is, the closest to a stair-like function.
            The smaller, the smoother the function.
            The parameter is clamped in [0 100]
        """
        super(Indexer, self).__init__()

        # store items
        if isinstance(items, torch.Tensor):
            self.register_parameter('items', nn.Parameter(items))
            self.items.requires_grad = items.requires_grad
        else:
            self.items = nn.ModuleList(items)

        self.dtype = dtype
        # remember parameters
        self.register_buffer(
            'cyclic', torch.as_tensor(cyclic))
        self.register_buffer(
            'n_items', torch.as_tensor(len(items)).type(self.dtype))
        self.register_buffer(
            '_stiffness', torch.as_tensor(stiffness).type(self.dtype))
        self._indices = None

    def _prepare_indices(self, indices):
        if self.cyclic:
            indices = torch.remainder(indices, 1.)
        else:
            indices = torch.clamp(indices, 0., 1.)
        return indices

    def inject_noise(self, indices):
        # change the value of the indices so as to still let them lead
        # to the same value
        # assuming the bins are valid (prepared)
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices).type(self.dtype)
        if indices.ndim==0:
            indices = indices[None]
        assert indices.ndim==1, 'indices must be a 1d tensor'

        delta = indices
        indices = self._prepare_indices(indices)
        delta = delta - indices

        bins = self.indices

        # having them go beyond [0, 1] to avoid boundary pbs 
        bins = torch.cat([k+bins for k in range(-1, 2)])

        closest = searchsorted(
            bins[None].contiguous(),
            indices[None].contiguous())[0]
        delta_plus = bins[closest] - indices
        delta_minus = indices - bins[closest-1]
        noise = torch.rand(*indices.shape, device=indices.device)
        noise = (delta_plus+delta_minus) * noise - delta_minus
        indices = indices + noise/3. + delta
        return indices

    def hard_forward(self, indices):
        #print('in hard forward with items', self.items,'\n      ', indices)
        """Internal (hard) forward function.
        This is the function that is called to build
        the output with fixed indices. 
        """
        # making indices between 0 and 1
        indices = self._prepare_indices(indices)

        # converting to indices in [0, n_items]
        indices = indices * self.n_items

        # getting the items to call (integer)
        closest_items = torch.min(
            input=torch.floor(indices),
            other=self.n_items-1).detach()

        # getting the (rescaled) indices for each one
        indices = indices - closest_items

        # pre-allocating the result
        if isinstance(self.items, torch.Tensor):
            # items are a tensor, we know the final shape
            out = torch.zeros(
                (len(indices), *self.items.shape[1:]),
                device=self.items.device
            )
        else:
            # otherwise we need to build results first
            out = None

        # looping over the items to call each one with its batch
        # of related indices
        closest_items  = closest_items.long()

        for item in range(int(self.n_items)):
            item_timesteps = torch.nonzero(
                closest_items==item,
                as_tuple=False
            )
            if not len(item_timesteps):
                continue
            item_timesteps = item_timesteps[..., 0]
            if isinstance(self.items, torch.Tensor):
                # if Tensor items, just take the corresponding entries
                out[item_timesteps] = self.items[item]
            else:
                # otherwise, call the item with rescaled indices
                result = self.items[item](
                    indices[item_timesteps]
                    + torch.finfo(self.dtype).eps)
                if out is None:
                    out = torch.zeros(
                        (len(indices), *result.shape[1:]),
                        device=result.device)
                out[item_timesteps] = result
        if out is None:
            print('None encountered. Breakpointing')
            import ipdb; ipdb.set_trace() 
        return out

    @property
    def stiffness(self):
        """get the stiffness parameter"""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value):
        """set the stiffness parameter, also to child nodes"""
        value = torch.as_tensor(
            value,
            device=self._stiffness.device,
            dtype=self.dtype
        )
        self._stiffness = value
        if isinstance(self.items, nn.ModuleList):
            for item in self.items:
                item.stiffness=value     

    @property
    def indices(self):
        """ get the indices for the whole tree. Only computed if
        required"""
        if self._indices is not None:
            return self._indices

        if isinstance(self.items, torch.Tensor):
            self._indices = torch.linspace(
                0,
                self.n_items-1, int(self.n_items),
                device=self.items.device,
                dtype=self.dtype
            )/self.n_items
        else:
            result = []
            pos = 0.
            delta = 1./self.n_items
            for index, item in enumerate(self.items):
                result += item.indices*delta + pos
                pos += delta
            self._indices = torch.stack(result)
        return self._indices

    def distance_to_midpoints(self, indices):
        # first make the indices tensor in the case they're not
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices).type(self.dtype).to(self.indices.device)
        if indices.ndim == 0:
            indices = indices[None]
        assert indices.ndim == 1, 'indices must be a 1d tensor'

        indices = self._prepare_indices(indices)
        # getting bins boundaries
        bins = self.indices
        bins = torch.cat((bins, torch.ones(1, dtype=bins.dtype, device=bins.device)))

        midpoints = 1/2 * (bins[:-1]+bins[1:])

        distance = torch.abs(indices[:, None]-midpoints[None,:])
        distance = torch.min(distance, dim=1).values
        return distance

    def forward(self, indices):
        """
        Application of the Indexer at specified indices. Will
        use the soft version if indices require grad.

        Parameters:
        indices: scalar, 0d or 1d Tensor
            the indices at which to retrieve the items.

        If the indices require grad, a smoothed approximation is used
        """

        # first make the indices tensor in the case they're not
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices).type(self.dtype)
        if indices.ndim==0:
            indices = indices[None]
        assert indices.ndim==1, 'indices must be a 1d tensor'

        if not indices.requires_grad:
            # if the indices don't require gradient, hard forward
            return self.hard_forward(indices)

        # clamping or cycling the indices
        indices = self._prepare_indices(indices)

        # getting bins boundaries
        bins = self.indices

        # computing midpoints
        if self.cyclic:
            bins = torch.cat((-1 + bins[-1,None], bins, 1.+ bins[:2]))
            midpoints = 1/2 * (bins[:-1]+bins[1:])
        else:
            bins = torch.cat((bins, 1.+ bins[:1]))

            midpoints = 1/2 * ( bins[:-1]+bins[1:])
            midpoints[0] = 0.
            midpoints[-1] = 1.

        # get values
        values = self.hard_forward(bins)
        values_shape =values.shape[1:]
        values = values.view(len(bins), -1)

        # getting the closest midpoints to each required index
        closest = searchsorted(midpoints[None].contiguous(),indices[None].contiguous())[0]
        if not self.cyclic:
            closest = torch.clamp(closest, 1., None)


        # computing the x-scale for applying the sigmoid
        # and computing the related scaled_input to the sigmoid
        # (so that we have 0 for closest bin, +-1 distance for closest midpoints)
        delta_x = torch.where(
            indices > bins[closest],
            torch.abs(midpoints[closest]-bins[closest]),
            torch.abs(midpoints[closest-1]-bins[closest])
        )
        scaled_input = (indices - bins[closest]) / torch.abs(delta_x)

        # computing the delta_y for each point (the jump of the stair there) 
        delta_y = values[closest] - values[closest-1]

        # the scale to apply to the logistic function so that it'll go from -1 to +1
        """stiffness = torch.rand(*indices.shape, device=indices.device) * 10 + 0.5
        scale = (1./(1 + torch.exp(-2*stiffness)) -
                 1./(1+torch.exp(2*stiffness)))
        scale = scale[:, None]
        """
        scale = (1./(1 + torch.exp(-2*self.stiffness)) -
                1./(1+torch.exp(2*self.stiffness)))

        # now compute the actual output
        out = (
            0.5 * (values[closest] + values[closest-1]) # central value
            + delta_y / scale
            * (
                #1/(1+torch.exp(-2*stiffness*scaled_input))[:, None]-0.5
                1/(1+torch.exp(-2*self.stiffness*scaled_input))[:, None]-0.5
            )
        )

        out = out.view(-1, *values_shape)
        return out
