define(['exports', 'module', 'react', 'classnames', './BootstrapMixin'], function (exports, module, _react, _classnames, _BootstrapMixin) {
  'use strict';

  var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

  function _interopRequire(obj) { return obj && obj.__esModule ? obj['default'] : obj; }

  var _React = _interopRequire(_react);

  var _classNames = _interopRequire(_classnames);

  var _BootstrapMixin2 = _interopRequire(_BootstrapMixin);

  var Button = _React.createClass({
    displayName: 'Button',

    mixins: [_BootstrapMixin2],

    propTypes: {
      active: _React.PropTypes.bool,
      disabled: _React.PropTypes.bool,
      block: _React.PropTypes.bool,
      navItem: _React.PropTypes.bool,
      navDropdown: _React.PropTypes.bool,
      componentClass: _React.PropTypes.node,
      href: _React.PropTypes.string,
      target: _React.PropTypes.string
    },

    getDefaultProps: function getDefaultProps() {
      return {
        bsClass: 'button',
        bsStyle: 'default',
        type: 'button'
      };
    },

    render: function render() {
      var classes = this.props.navDropdown ? {} : this.getBsClassSet();
      var renderFuncName = undefined;

      classes = _extends({
        active: this.props.active,
        'btn-block': this.props.block }, classes);

      if (this.props.navItem) {
        return this.renderNavItem(classes);
      }

      renderFuncName = this.props.href || this.props.target || this.props.navDropdown ? 'renderAnchor' : 'renderButton';

      return this[renderFuncName](classes);
    },

    renderAnchor: function renderAnchor(classes) {

      var Component = this.props.componentClass || 'a';
      var href = this.props.href || '#';
      classes.disabled = this.props.disabled;

      return _React.createElement(
        Component,
        _extends({}, this.props, {
          href: href,
          className: (0, _classNames)(this.props.className, classes),
          role: 'button' }),
        this.props.children
      );
    },

    renderButton: function renderButton(classes) {
      var Component = this.props.componentClass || 'button';

      return _React.createElement(
        Component,
        _extends({}, this.props, {
          className: (0, _classNames)(this.props.className, classes) }),
        this.props.children
      );
    },

    renderNavItem: function renderNavItem(classes) {
      var liClasses = {
        active: this.props.active
      };

      return _React.createElement(
        'li',
        { className: (0, _classNames)(liClasses) },
        this.renderAnchor(classes)
      );
    }
  });

  module.exports = Button;
});
// eslint-disable-line object-shorthand