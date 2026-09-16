from sqlalchemy import Boolean, ForeignKey, Integer, String, JSON, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db import Base


class Restaurant(Base):

    __tablename__ = "restaurants"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    verification_token: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True
    )

    menus = relationship(
        "Menu",
        back_populates="restaurant",
        cascade="all, delete-orphan"
    )


class Menu(Base):

    __tablename__ = "menus"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    restaurant_id: Mapped[int] = mapped_column(
        ForeignKey("restaurants.id"),
        nullable=False,
        index=True
    )

    menu_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    menu_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )

    status: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    restaurant = relationship(
        "Restaurant",
        back_populates="menus"
    )

    images = relationship(
        "MenuImage",
        back_populates="menu",
        cascade="all, delete-orphan"
    )

    menu_items = relationship(
        "MenuItem",
        back_populates="menu",
        cascade="all, delete-orphan"
    )

    menu_items_complete = relationship(
        "MenuItemComplete",
        back_populates="menu",
        cascade="all, delete-orphan"
    )


class MenuImage(Base):

    __tablename__ = "menu_images"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    menu_id: Mapped[int] = mapped_column(
        ForeignKey("menus.id"),
        nullable=False,
        index=True
    )

    image_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False
    )

    menu = relationship(
        "Menu",
        back_populates="images"
    )

    page_coordinates = relationship(
        "PageCoordinates",
        back_populates="menu_image",
        cascade="all, delete-orphan"
    )

    menu_items = relationship(
        "MenuItem",
        back_populates="menu_image"
    )
    menu_items_complete = relationship(       # <-- YE NAYI LINES ADD KARO
        "MenuItemComplete",
        back_populates="menu_image"
    )


class PageCoordinates(Base):

    __tablename__ = "page_coordinates"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    menu_image_id: Mapped[int] = mapped_column(
        ForeignKey("menu_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    matched_text: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    raw_text: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True
    )

    detected_label: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    source_image: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    source_image_path: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    photo_bbox: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    image_file: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )

    menu_image = relationship(
        "MenuImage",
        back_populates="page_coordinates"
    )


class MenuItem(Base):

    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    # Menu relationship
    menu_id: Mapped[int] = mapped_column(
        ForeignKey("menus.id"),
        nullable=False,
        index=True
    )

    # Menu image relationship (jis page se ye item aya - agar koi photo
    # linked hui to usi page ka menu_image_id)
    menu_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("menu_images.id"),
        nullable=True,
        index=True
    )

    product_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    arabic_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    unit: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    has_photo_on_menu: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    image_file: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    # Confidence scores (JSON ke "confidence" object se aate hain)
    product_name_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    price_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    arabic_name_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    source: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )
    # Relationships
    variants = relationship(
        "MenuItemVariant",
        back_populates="menu_item",
        cascade="all, delete-orphan"
    )

    menu = relationship(
        "Menu",
        back_populates="menu_items"
    )

    menu_image = relationship(
        "MenuImage",
        back_populates="menu_items"
    )


class MenuItemVariant(Base):

    __tablename__ = "menu_item_variants"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    menu_item_id: Mapped[int] = mapped_column(
        ForeignKey("menu_items.id"),
        nullable=False,
        index=True
    )

    size: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    menu_item = relationship(
        "MenuItem",
        back_populates="variants"
    )


class MenuItemComplete(Base):

    __tablename__ = "menu_items_complete"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    # Menu relationship
    menu_id: Mapped[int] = mapped_column(
        ForeignKey("menus.id"),
        nullable=False,
        index=True
    )

    # Menu image relationship (jis page se ye item aya - agar koi photo
    # linked hui to usi page ka menu_image_id)
    menu_image_id: Mapped[int | None] = mapped_column(
        ForeignKey("menu_images.id"),
        nullable=True,
        index=True
    )

    product_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    arabic_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    unit: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    has_photo_on_menu: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    image_file: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    # Confidence scores (JSON ke "confidence" object se aate hain)
    product_name_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    price_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    arabic_name_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Relationships
    variants = relationship(
        "MenuItemCompleteVariant",
        back_populates="menu_item",
        cascade="all, delete-orphan"
    )

    source: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )

    menu = relationship(
        "Menu",
        back_populates="menu_items_complete"
    )

    menu_image = relationship(
        "MenuImage",
        back_populates="menu_items_complete"
    )


class MenuItemCompleteVariant(Base):

    __tablename__ = "menu_item_complete_variants"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    menu_item_id: Mapped[int] = mapped_column(
        ForeignKey("menu_items_complete.id"),
        nullable=False,
        index=True
    )

    size: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    menu_item = relationship(
        "MenuItemComplete",
        back_populates="variants"
    )
