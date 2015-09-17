define(['exports', 'module', 'react', './Button', './FormGroup', './InputBase'], function (exports, module, _react, _Button, _FormGroup, _InputBase2) {
  'use strict';

  var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

  var _createClass = (function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ('value' in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; })();

  function _interopRequire(obj) { return obj && obj.__esModule ? obj['default'] : obj; }

  function _objectWithoutProperties(obj, keys) { var target = {}; for (var i in obj) { if (keys.indexOf(i) >= 0) continue; if (!Object.prototype.hasOwnProperty.call(obj, i)) continue; target[i] = obj[i]; } return target; }

  function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError('Cannot call a class as a function'); } }

  function _inherits(subClass, superClass) { if (typeof superClass !== 'function' && superClass !== null) { throw new TypeError('Super expression must either be null or a function, not ' + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) subClass.__proto__ = superClass; }

  var _React = _interopRequire(_react);

  var _Button2 = _interopRequire(_Button);

  var _FormGroup2 = _interopRequire(_FormGroup);

  var _InputBase3 = _interopRequire(_InputBase2);

  function valueValidation(_ref, propName, componentName) {
    var children = _ref.children;
    var value = _ref.value;

    if (children && value) {
      return new Error('Both value and children cannot be passed to ButtonInput');
    }
    return _React.PropTypes.oneOfType([_React.PropTypes.string, _React.PropTypes.number]).call(null, { children: children, value: value }, propName, componentName);
  }

  var ButtonInput = (function (_InputBase) {
    function ButtonInput() {
      _classCallCheck(this, ButtonInput);

      if (_InputBase != null) {
        _InputBase.apply(this, arguments);
      }
    }

    _inherits(ButtonInput, _InputBase);

    _createClass(ButtonInput, [{
      key: 'renderFormGroup',
      value: function renderFormGroup(children) {
        var _props = this.props;
        var bsStyle = _props.bsStyle;
        var value = _props.value;

        var other = _objectWithoutProperties(_props, ['bsStyle', 'value']);

        // eslint-disable-line object-shorthand, no-unused-vars
        return _React.createElement(
          _FormGroup2,
          other,
          children
        );
      }
    }, {
      key: 'renderInput',
      value: function renderInput() {
        var _props2 = this.props;
        var children = _props2.children;
        var value = _props2.value;

        var other = _objectWithoutProperties(_props2, ['children', 'value']);

        // eslint-disable-line object-shorthand
        var val = children ? children : value;
        return _React.createElement(_Button2, _extends({}, other, { componentClass: 'input', ref: 'input', key: 'input', value: val }));
      }
    }]);

    return ButtonInput;
  })(_InputBase3);

  ButtonInput.defaultProps = {
    type: 'button'
  };

  ButtonInput.propTypes = {
    type: _React.PropTypes.oneOf(['button', 'reset', 'submit']),
    bsStyle: function bsStyle(props) {
      //defer to Button propTypes of bsStyle
      return null;
    },
    children: valueValidation,
    value: valueValidation
  };

  module.exports = ButtonInput;
});