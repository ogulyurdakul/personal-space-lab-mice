var MargMeans = function(dist){
  return map( function(key){
    return expectation(marginalize(dist, function(x){return x[key]}))
  }, _.keys(dist.samples[0].value))}

var MargStds = function(dist, means){
  return map(function(x){Math.pow(x, 0.5)}, map2(
    function(key, mean){return expectation(marginalize(dist, function(x){return x[key]}),
                                           function(x){ return Math.pow(x - mean,2) }
                                          )
                       }, _.keys(dist.samples[0].value), means))}

var sequentialInference = function(prior, data, totalDataLength){

  var curInferred = prior
  var totalDataLength = (totalDataLength <= 0) ? data.length : totalDataLength

  if (data.length == 0) {
    print('Done!')
    return curInferred
  } else {
    print((data.length) + ' data points remaining out of ' + totalDataLength)
    var posterior = oneStepMouse(prior, data[0])
    return sequentialInference(posterior, _.drop(data,1), totalDataLength)
  }
}

var oneStepMouse = function(prior, datum){

  var posterior = Infer(function(){

    var params = {atk_x: gaussian({mu: prior.atk_x.mean, 
                                   sigma: prior.atk_x.std}),
                  atk_dx: gaussian({mu: prior.atk_dx.mean, 
                                    sigma: prior.atk_dx.std}),
                  inv_x: gaussian({mu: prior.inv_x.mean, 
                                   sigma: prior.inv_x.std}),
                  inv_dx: gaussian({mu: prior.inv_dx.mean, 
                                    sigma: prior.inv_dx.std})}

    var transition = dp.cache(function(state){ // the state is [x, dx]
      var dt = 1/30
      return [state[0] + state[1]*dt, state[1]]
    })

    var actionUtilities = function(action, state){
      if (action == 'attack') {return params.atk_x * state[0] + params.atk_dx * state[1]
                           } 
      else if (action == 'investigation') {return params.inv_x * state[0] + params.inv_dx * state[1]
                                }
      else if (action == 'none') {
        return 0
      }
    }

    var recursiveDecision = dp.cache(function(actions, state, gamma, depth){
      if (depth == 0) {
        var maxUtilities = maxWith(function(action){
          actionUtilities(action, state)
        }, actions)
        return maxUtilities
      } else {
        var maxUtilities = maxWith(function(action){
          actionUtilities(action, state) + recursiveDecision(actions,
                                                             transition(state),
                                                             gamma,
                                                             depth-1)[1]
        }, actions)
        return maxUtilities
      }
    })

    var actions = ['attack', 'investigation', 'none']
    var gamma = 0.9
    var maxDepth = 25

    condition(datum.action == recursiveDecision(actions, 
                                                datum.state,
                                                gamma, 
                                                maxDepth)[0])

    return params

  })

  var means = MargMeans(posterior)
  var stds = MargStds(posterior, means)

  return {atk_x: {mean: means[0],
                  std: stds[0]},
          atk_dx: {mean: means[1],
                   std: stds[1]},
          inv_x: {mean: means[2],
                  std: stds[2]},
          inv_dx: {mean: means[3],
                   std: stds[3]},
         }

}

var data = [
  {state: [4,  0], action: 'none'},
  {state: [4,  0], action: 'none'},
  {state: [4,  1], action: 'investigation'},
  {state: [3,  2], action: 'investigation'},
  {state: [1,  0], action: 'attack'},
  {state: [1,  0], action: 'attack'},
  {state: [1, -1], action: 'attack'},
  {state: [2, -1], action: 'none'},
  {state: [3, -1], action: 'none'},
  {state: [4,  0], action: 'none'}
]

var prior = {atk_x: {mean: 0, std: 2}, 
             atk_dx: {mean: 0, std: 2},
             inv_x: {mean: 0, std: 2}, 
             inv_dx: {mean: 0, std: 2}}

var learnedUtility = sequentialInference(prior, data, 0)
print(learnedUtility)