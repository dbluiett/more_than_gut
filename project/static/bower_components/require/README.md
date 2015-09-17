Require.js
=======

commonjs require loader and compiler [http://nathanfaucett.github.io/require.js/](http://nathanfaucett.github.io/require.js/)

```
// install the package
$ npm install require.js
```


## Usage
add a script tag to your html file and point it towards the require.js file, then set a data-main property to your javascript main file
```html
<script src="path/to/require.js" data-main="path/to/main.js"></script>
```

###Attribute Options

  - main - file path to load, if this and global attributes not present will attach require and module to global
  - global - will attach require and module to global object
  - env - list separated by spaces of environment variables, ex "NODE_ENV=development DEBUG=*"
  - argv - list separated by spaces of argv variables, ex "--testing true"
  
###Compile Options

install globally
```
$ npm install -g require.js
```

example
```
$ requirejs -f /path/to/index.js -o /path/to/out.js --exportName globalName
```

###Options:
  - -f, --file        index file to start parsing from                                               [required]
  - -o, --out         out file, defaults to index file + min.js                                    
  - -e, --exportName  export to global object with this name                                       
  - -a, --argv        list of arguments to pass to process.argv (--argv=--arg0,-a)                 
  - --env             list of arguments to pass to process.env (--env=NODE_ENV=development,DEBUG=*)
  - -v, --verbose     verbose mode