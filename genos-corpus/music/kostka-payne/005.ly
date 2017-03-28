\version "2.12.0"

global = { 
  \key d \major 
  \time 4/2 
  \partial 2
}

letra = \lyricmode { 
  Ach lie -- ber Her -- re Je -- su Christ, weil du ein Kind ge -- we -- sen bist, 
  so gib auch die -- sem Kin -- de -- lein dein Gnad und auch den Se -- gen dein; 
  ach Je -- sus, Her -- re mein, be -- hüt dies Kin -- de -- lein. 
} 

\header {
  title = "Ach lieber Herre Jesu Christ"
  composer = "Brahms"
}

\new ChoirStaff <<
  \new Staff <<
    \set Staff.instrumentName = \markup {"Soprano"}
    \global
    \relative c'' {
      d2^\p
      cis b cis^\< d
      b^\> a\! a d^\<
      cis^\> b\! fis a
      g fis fis e
      fis g g^\< fis
      e^\> d\! d a'
      a b cis^\< d
      a^\> fis\! fis e^\p^\<
      g^\> fis\! d(^\< fis
      g)^\> fis\! fis e^\pp^\<
      g^\>^\markup {\italic "dim. rit."} fis\! d(^\< fis
      g)^\> fis\! fis\fermata
      \bar "|."
    }
  >>
  \addlyrics { \letra }

  \new Staff <<
    \set Staff.instrumentName = \markup {"Alto"}
    \global
    \relative c'' {
      b2^\p
      a g a^\< a
      g^\> e\! e fis^\<
      fis^\> d\! cis fis
      d d d cis
      d d d^\< d 
      b^\> b\! b cis
      e g g^\< a
      e^\> d\! d cis^\p^\<
      d^\> cis\! b(^\< cis
      d)^\> d\! d cis^\pp^\<
      e^\>^\markup {\italic "dim. rit."} cis\! b(^\< cis
      d)^\> d\! d\fermata
      \bar "|."
    }
  >>
  \addlyrics { \letra }

  \new Staff <<
    \set Staff.instrumentName = \markup {"Tenor"}
    \global
    \clef "G_8"
    \relative c' {
      b2^\p
      e d e^\< fis
      d^\> cis\! cis a^\<
      a^\> b\! cis cis
      b a a a 
      a b b^\< a
      g^\> fis\! fis a
      cis d e^\< fis
      e^\> a, a a^\p^\<
      b^\> cis\! d(^\< cis
      b)^\> a\! a a^\pp^\<
      b^\>^\markup {\italic "dim. rit."} a\! d(^\< cis 
      b)^\> a\! a\fermata
      \bar "|."
    }
  >>
  \addlyrics { \letra }

  \new Staff <<
    \set Staff.instrumentName = \markup {"Bass"}
    \global
    \clef bass
    \relative c' { 
      g2^\p
      a b a^\< fis
      g^\> a\! a d,^\<
      fis^\> g\! a fis
      g d d a'
      d, g b,^\< d
      e^\> b\! b a
      a' g e^\< d
      cis^\> d\! d a'^\p^\<
      g^\> a\! b(^\< a
      g)^\> d\! d a'^\pp^\<
      e^\>^\markup {\italic "dim. rit."} fis\! b(^\< a
      g)^\> d\! d\fermata
      \bar "|."
    }
  >>
  \addlyrics { \letra }

>>