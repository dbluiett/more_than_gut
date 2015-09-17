define(['exports', 'module', 'react', 'classnames', './BootstrapMixin', './CollapsibleMixin', './utils/deprecatedProperty'], function (exports, module, _react, _classnames, _BootstrapMixin, _CollapsibleMixin, _utilsDeprecatedProperty) {
  'use strict';

  var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

  function _interopRequire(obj) { return obj && obj.__esModule ? obj['default'] : obj; }

  var _React = _interopRequire(_react);

  var _classNames = _interopRequire(_classnames);

  var _BootstrapMixin2 = _interopRequire(_BootstrapMixin);

  var _CollapsibleMixin2 = _interopRequire(_CollapsibleMixin);

  var _collapsable = _interopRequire(_utilsDeprecatedProperty);

  var Panel = _React.createClass({
    displayName: 'Panel',

    mixins: [_BootstrapMixin2, _CollapsibleMixin2],

    propTypes: {
      collapsable: _collapsable,
      collapsible: _React.PropTypes.bool,
      onSelect: _React.PropTypes.func,
      header: _React.PropTypes.node,
      id: _React.PropTypes.string,
      footer: _React.PropTypes.node,
      eventKey: _React.PropTypes.any
    },

    getDefaultProps: function getDefaultProps() {
      return {
        bsClass: 'panel',
        bsStyle: 'default'
      };
    },

    handleSelect: function handleSelect(e) {
      e.selected = true;

      if (this.props.onSelect) {
        this.props.onSelect(e, this.props.eventKey);
      } else {
        e.preventDefault();
      }

      if (e.selected) {
        this.handleToggle();
      }
    },

    handleToggle: function handleToggle() {
      this.setState({ expanded: !this.state.expanded });
    },

    getCollapsibleDimensionValue: function getCollapsibleDimensionValue() {
      return _React.findDOMNode(this.refs.panel).scrollHeight;
    },

    getCollapsibleDOMNode: function getCollapsibleDOMNode() {
      if (!this.isMounted() || !this.refs || !this.refs.panel) {
        return null;
      }

      return _React.findDOMNode(this.refs.panel);
    },

    render: function render() {
      var classes = this.getBsClassSet();
      var collapsible = this.props.collapsible || this.props.collapsable;

      return _React.createElement(
        'div',
        _extends({}, this.props, {
          className: (0, _classNames)(this.props.className, classes),
          id: collapsible ? null : this.props.id, onSelect: null }),
        this.renderHeading(),
        collapsible ? this.renderCollapsableBody() : this.renderBody(),
        this.renderFooter()
      );
    },

    renderCollapsableBody: function renderCollapsableBody() {
      var collapseClass = this.prefixClass('collapse');

      return _React.createElement(
        'div',
        {
          className: (0, _classNames)(this.getCollapsibleClassSet(collapseClass)),
          id: this.props.id,
          ref: 'panel',
          'aria-expanded': this.isExpanded() ? 'true' : 'false' },
        this.renderBody()
      );
    },

    renderBody: function renderBody() {
      var allChildren = this.props.children;
      var bodyElements = [];
      var panelBodyChildren = [];
      var bodyClass = this.prefixClass('body');

      function getProps() {
        return { key: bodyElements.length };
      }

      function addPanelChild(child) {
        bodyElements.push((0, _react.cloneElement)(child, getProps()));
      }

      function addPanelBody(children) {
        bodyElements.push(_React.createElement(
          'div',
          _extends({ className: bodyClass }, getProps()),
          children
        ));
      }

      function maybeRenderPanelBody() {
        if (panelBodyChildren.length === 0) {
          return;
        }

        addPanelBody(panelBodyChildren);
        panelBodyChildren = [];
      }

      // Handle edge cases where we should not iterate through children.
      if (!Array.isArray(allChildren) || allChildren.length === 0) {
        if (this.shouldRenderFill(allChildren)) {
          addPanelChild(allChildren);
        } else {
          addPanelBody(allChildren);
        }
      } else {

        allChildren.forEach((function (child) {
          if (this.shouldRenderFill(child)) {
            maybeRenderPanelBody();

            // Separately add the filled element.
            addPanelChild(child);
          } else {
            panelBodyChildren.push(child);
          }
        }).bind(this));

        maybeRenderPanelBody();
      }

      return bodyElements;
    },

    shouldRenderFill: function shouldRenderFill(child) {
      return _React.isValidElement(child) && child.props.fill != null;
    },

    renderHeading: function renderHeading() {
      var header = this.props.header;
      var collapsible = this.props.collapsible || this.props.collapsable;

      if (!header) {
        return null;
      }

      if (!_React.isValidElement(header) || Array.isArray(header)) {
        header = collapsible ? this.renderCollapsableTitle(header) : header;
      } else {
        var className = (0, _classNames)(this.prefixClass('title'), header.props.className);

        if (collapsible) {
          header = (0, _react.cloneElement)(header, {
            className: className,
            children: this.renderAnchor(header.props.children)
          });
        } else {
          header = (0, _react.cloneElement)(header, { className: className });
        }
      }

      return _React.createElement(
        'div',
        { className: this.prefixClass('heading') },
        header
      );
    },

    renderAnchor: function renderAnchor(header) {
      return _React.createElement(
        'a',
        {
          href: '#' + (this.props.id || ''),
          className: this.isExpanded() ? null : 'collapsed',
          'aria-expanded': this.isExpanded() ? 'true' : 'false',
          onClick: this.handleSelect },
        header
      );
    },

    renderCollapsableTitle: function renderCollapsableTitle(header) {
      return _React.createElement(
        'h4',
        { className: this.prefixClass('title') },
        this.renderAnchor(header)
      );
    },

    renderFooter: function renderFooter() {
      if (!this.props.footer) {
        return null;
      }

      return _React.createElement(
        'div',
        { className: this.prefixClass('footer') },
        this.props.footer
      );
    }
  });

  module.exports = Panel;
});