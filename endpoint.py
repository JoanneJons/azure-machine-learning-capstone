import requests
import json

scoring_uri = 'http://a6d9f881-3dba-4b47-a901-a13f96fef301.southcentralus.azurecontainer.io/score'
key = 'wLNKqhKi4Fr2f0vsmMfMxnPoCPMgzrQz'

data = {
    "data":[
        {
            "Column1": 204,
            "Column1_1": 204,
            "radius_mean": 12.47,
            "texture_mean": 18.6,
            "perimeter_mean": 81.09,
            "area_mean": 481.9,
            "smoothness_mean": 0.09965,
            "compactness_mean": 0.1058,
            "concavity_mean": 0.08005,
            "concave points_mean": 0.03821,
            "symmetry_mean": 0.1925,
            "fractal_dimension_mean": 0.06373,
            "radius_se": 0.3961,
            "texture_se": 1.044,
            "perimeter_se": 2.497,
            "area_se": 30.29,
            "smoothness_se": 0.006953,
            "compactness_se": 0.01911,
            "concavity_se": 0.02701,
            "concave points_se": 0.01037,
            "symmetry_se": 0.01782,
            "fractal_dimension_se": 0.003586,
            "radius_worst": 14.97,
            "texture_worst": 24.64,
            "perimeter_worst": 96.05,
            "area_worst": 677.9,
            "smoothness_worst": 0.1426,
            "compactness_worst": 0.2378,
            "concavity_worst": 0.2671,
            "concave points_worst": 0.1015,
            "symmetry_worst": 0.3014,
            "fractal_dimension_worst": 0.0875
        },
        {
            "Column1": 70,
            "Column1_1": 70,
            "radius_mean": 18.94,
            "texture_mean": 21.31,
            "perimeter_mean": 123.6,
            "area_mean": 1130,
            "smoothness_mean": 0.09009,
            "compactness_mean": 0.1029,
            "concavity_mean": 0.108,
            "concave points_mean": 0.07951,
            "symmetry_mean": 0.1582,
            "fractal_dimension_mean": 0.05461,
            "radius_se": 0.7888,
            "texture_se": 0.7975,
            "perimeter_se": 5.486,
            "area_se": 96.05,
            "smoothness_se": 0.004444,
            "compactness_se": 0.01652,
            "concavity_se": 0.02269,
            "concave points_se": 0.0137,
            "symmetry_se": 0.01386,
            "fractal_dimension_se": 0.001698,
            "radius_worst": 24.86,
            "texture_worst": 26.58,
            "perimeter_worst": 165.9,
            "area_worst": 1866,
            "smoothness_worst": 0.1193,
            "compactness_worst": 0.2336,
            "concavity_worst": 0.2687,
            "concave points_worst": 0.1789,
            "symmetry_worst": 0.2551,
            "fractal_dimension_worst": 0.06589            
        }
    ]
}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
