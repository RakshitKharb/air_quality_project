import requests
import pandas as pd

def fetch_parameter_data(param, city, start_date, end_date, headers):
    base_url = "https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "parameter": param,
        "date_from": start_date,
        "date_to": end_date,
        "limit": 10000,
        "page": 1
    }
    all_data = []
    while True:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            for result in results:
                if 'date' in result and 'utc' in result['date']:
                    all_data.append({
                        'datetime': result['date']['utc'],
                        'parameter': result['parameter'],
                        'value': result['value']
                    })
            params['page'] += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {param}: {e}")
            break
    return all_data

def fetch_openaq_data_parallel(city, parameters, start_date, end_date, api_key):
    import concurrent.futures
    headers = {"X-API-Key": api_key}
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_parameter_data, param, city, start_date, end_date, headers) for param in parameters]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            all_data.extend(data)
    if all_data:
        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.pivot_table(values='value', index='datetime', columns='parameter', aggfunc='mean')
        df = df.resample('H').mean()
        return df
    else:
        return None
