# Configure the Azure Provider
provider "azurerm" {
  # whilst the `version` attribute is optional, we recommend pinning to a given version of the Provider
  version = "=2.0.0"
  features {}
}

# Create a resource group
resource "sohail_res_group2" "example" {
  name     = "example-resources"
  location = "North Europe"
}

# Create a virtual network within the resource group
resource "azurerm_virtual_network" "example" {
  name                = "example-network"
  resource_group_name = sohail_res_group2.example.name
  location            = sohail_res_group2.example.location
  address_space       = ["10.0.0.0/16"]
}

