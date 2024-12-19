import pickle
import requests
from urllib.parse import urlparse, parse_qs
import re
import ipaddress

# Feature extraction class
class Extractor:
    def __init__(self):
        pass

    def num_dots(self, url):
        return url.count('.')

    def subdomain_level(self, url):
        parsed_url = urlparse(url)
        if parsed_url.hostname:
            subdomains = parsed_url.hostname.split('.')
            return len(subdomains)
        else:
            subdomains = parsed_url.path.split('.')
            return len(subdomains)
        
    def path_level(self, url):
        path = urlparse(url).path
        return path.count('/') + 1  # Add 1 to account for the root path
    
    def url_length(self, url):
        return len(url)
    
    def num_dash(self, url):
        return url.count('-')
    
    def num_dash_in_hostname(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('-') + parsed_url.path.count('-')
    
    def at_symbol(self, url):
        parsed_url = urlparse(url)
        return url.count('@')
    
    def tilde_symbol(self, url):
        parsed_url = urlparse(url)
        return url.count('~')

    def num_underscore(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('_') + parsed_url.path.count('_') + parsed_url.query.count('_') + parsed_url.fragment.count('_') 
    
    def num_percent(self, url):
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('%') + parsed_url.path.count('%') + parsed_url.query.count('%') + parsed_url.fragment.count('%') 

    def num_query_components(self, url):
        parsed_url = urlparse(url)
        return len(parse_qs(parsed_url.query))
    
    def num_ampersand(self, url):
        return url.count('&')
    
    def num_hash(self, url):
        return url.count('#')
    
    def num_numeric_chars(self, url):
        return sum(c.isdigit() for c in url)
    
    def no_https(self, url):
        parsed_url = urlparse(url)
        if (parsed_url.scheme=="https" or parsed_url.scheme==""):
            return 0
        else:
            return 1
        

    def random_string(self, url):
        # Check if the URL contains a random-looking string
        return 1 if bool(re.search(r'[0-9a-f]{8}[-]?([0-9a-f]{4}[-]?){3}[0-9a-f]{12}', url, re.I)) else 0
        
    
    #listing shortening services
    shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                        r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                        r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                        r"tr\.im|link\.zip\.net"
    
    def tinyURL(self, url):
        shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                        r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                        r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                        r"tr\.im|link\.zip\.net"
        
        match=re.search(shortening_services,url)
        if match:
            return 1
        else:
            return 0
    


    def https_in_hostname(self, url):
        parsed_url = urlparse(url)
        return 1 if 'https' in parsed_url.netloc or 'https' in parsed_url.path else 0
    
    def hostname_length(self, url):
        parsed_url = urlparse(url)
        if parsed_url.hostname:
            return len(parsed_url.hostname)
        else: 
            return len(parsed_url.path.split('/')[0])

    def path_length(self, url):
        parsed_url = urlparse(url)
        if parsed_url.hostname:
            return len(parsed_url.path)
        else: 
            return (len(parsed_url.path.split('/', 1)[-1])+1)

    def query_length(self, url):
        return len(urlparse(url).query)


    def double_slash_in_path(self, url):
        return 1 if '//' in urlparse(url).path else 0
    

    def ip_address(url):
    
        url_domain = re.search(r'(?:https?://)?([^/]+)', url).group(1)

        # Define regular expressions
        ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        ipv4_pattern = re.compile(r'(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[0-9]?[0-9])(\.|$){4}')
        hex_ipv4_pattern = re.compile(r'(0x([0-9][0-9]|[A-F][A-F]|[A-F][0-9]|[0-9][A-F]))(\.|$){4}')

        # Check if the URL domain matches any pattern
        if ip_pattern.match(url_domain) or ipv4_pattern.match(url_domain) or hex_ipv4_pattern.match(url_domain):
            result = 1
        else:
            result = 0
            
        return result


    def iframe(self, response):
        if response == "":
            return 1
        else:
            if re.findall(r"[<iframe>|<frameBorder>]", response.text):
                return 0
            else:
                return 1
            

    def mouseOver(self, response): 
        if response == "" :
            return 1
        else:
            if re.findall("<script>.+onmouseover.+</script>", response.text):
                return 1
            else:
                return 0
            
    def forwarding(self, response):
        if response == "":
            return 1
        else:
            if len(response.history) <= 2:
                return 0
            else:
                return 1
            


    def extract(self, url):
        features = []


        features.append(self.num_dots(url))
        features.append(self.subdomain_level(url))
        features.append(self.path_level(url))
        features.append(self.url_length(url))
        features.append(self.num_dash(url))
        features.append(self.num_dash_in_hostname(url))
        features.append(self.at_symbol(url))
        features.append(self.tilde_symbol(url))
        features.append(self.num_underscore(url))
        features.append(self.num_percent(url))
        features.append(self.num_query_components(url))
        features.append(self.num_ampersand(url))
        features.append(self.num_hash(url))
        features.append(self.num_numeric_chars(url))
        features.append(self.no_https(url))
        features.append(self.random_string(url))
        features.append(self.https_in_hostname(url))
        features.append(self.hostname_length(url))
        features.append(self.path_length(url))
        features.append(self.query_length(url))
        features.append(self.double_slash_in_path(url))
        features.append(self.tinyURL(url))

        try:
            response = requests.get(url)
        except:
            response = ""

        features.append(self.iframe(response))    
        features.append(self.mouseOver(response))
        features.append(self.forwarding(response))

        return features
